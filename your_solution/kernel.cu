#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// CUTLASS: used for the RowMajorTensorOpMultiplicandCrosswise SMEM layout
// (canonical ldmatrix-friendly INT4 swizzle) and a typed wrapper around the
// sm_80 INT4 MMA PTX. Build requires v4.4.2+ headers on
// NVCC_APPEND_FLAGS="-I/path/to/cutlass/include".
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/arch/mma.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/layout/tensor_op_multiplicand_sm80.h>

using CutlassMmaOp = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 64>,
    32,
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int,              cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAddSaturate
>;

// ============================================================================
// Offline-quant kernel: writes block-major A / scales_A for the GEMM consumer.
// QUANT_BLOCK_M MUST match the GEMM's BLOCK_M. v3 drops it from 128 -> 64.
// ============================================================================
__global__ void quantize_int4_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int M,
    int K,
    int group_size)
{
    constexpr int QUANT_BLOCK_M = 64;   // MUST match GEMM's BLOCK_M (v3)

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;
    if (row >= M) return;

    int num_groups = K / group_size;
    int k_start = group * group_size;
    int half_gs = group_size / 2;

    int m_block = row / QUANT_BLOCK_M;
    int row_in_block = row - m_block * QUANT_BLOCK_M;

    float max_abs = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float v = __half2float(input[row * K + k_start + i]);
        float a = fabsf(v);
        if (a > max_abs) max_abs = a;
    }

    float scale = max_abs / 7.0f;
    size_t scale_off = (size_t)m_block * num_groups * QUANT_BLOCK_M
                     + (size_t)group * QUANT_BLOCK_M
                     + (size_t)row_in_block;
    scales[scale_off] = __float2half(scale);

    float rscale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    size_t out_base = (size_t)m_block * num_groups * QUANT_BLOCK_M * half_gs
                    + (size_t)group * QUANT_BLOCK_M * half_gs
                    + (size_t)row_in_block * half_gs;
    for (int i = 0; i < group_size; i += 2) {
        float ve = __half2float(input[row * K + k_start + i]);
        float vo = __half2float(input[row * K + k_start + i + 1]);
        int qe = __float2int_rn(ve * rscale);
        int qo = __float2int_rn(vo * rscale);
        qe = max(-8, min(7, qe));
        qo = max(-8, min(7, qo));
        uint8_t packed = (uint8_t)((qo & 0xF) << 4) | (uint8_t)(qe & 0xF);
        output[out_base + i / 2] = packed;
    }
}

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    auto output = torch::empty({M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    int num_groups = K / group_size;
    auto scales = torch::empty({M, num_groups},
        torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 block(256);
    dim3 grid((M + 255) / 256, num_groups);
    quantize_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size);
    return {output, scales};
}

// ============================================================================
// GEMM kernel -- COMET-style W4A4, v3: tile-shape restructure for occupancy
//
// v2 (199.5 baseline) was pinned at 2 blocks/SM with 8 warps/block = 16 active
// warps/SM. The MMA pipe was 65% utilized but we could not break past it
// because there were simply not enough live warps to hide dependency/issue
// latency in the mainloop. The scale-FMA work inside the K loop cost too much
// when only 16 warps competed for issue slots.
//
// v3 attacks that wall by shrinking every live-state dimension:
//   BLOCK_M=BLOCK_N=64 (was 128 each)   -> 1/4 the output area per CTA
//   NUM_WARPS=4 in a 2x2 grid           -> 1/2 the warps per CTA
//   TILES_M=TILES_N=2 (was 2 x 8)       -> 1/4 the MMA count per K-iter
//   acc[2][2][4] = 16 fp32/thread       -> 1/4 the accumulator registers
//
// Expected register envelope: ~55-65 regs/thread (was ~100). That puts us at
// 8 blocks/SM from the register limit, 32 active warps/SM -- 2x the old
// latency-hiding budget. Per-SM MMA work rate is unchanged (more CTAs, each
// doing less), so the win is purely in pipeline/issue saturation.
//
// CTA count grows 4x (stronger parallelism on 84 SMs), but per-CTA epilogue
// work shrinks 4x, so launch/tail overhead stays bounded.
//
// SMEM per stage  : 64*32 + 64*32 + 64*2 + 64*2 = 4352 B
// SMEM per block  : 2 stages -> 8704 B   (prev. 17408)
// SMEM occupancy  : 96000 / 8704 = 11 blocks/SM -- not the limit
// Register limit  : 65536 / (128 * 64 regs) ~= 8 blocks/SM -- the limit
//
// COMET-direction invariants preserved:
//   - Block-major A/B/scales layouts (unchanged)
//   - scales_A / scales_B in the same cp.async group as the weight tiles
//   - 2 __syncthreads / K-iter (top + end); no smem_sb staging
//   - CUTLASS crosswise SMEM swizzle for ldmatrix-friendly weights
// ============================================================================

static constexpr int BLOCK_M    = 64;
static constexpr int BLOCK_N    = 64;
static constexpr int BLOCK_K    = 64;     // must equal quant group_size
static constexpr int WARP_SZ    = 32;
static constexpr int NUM_WARPS  = 4;
static constexpr int WARPS_M    = 2;
static constexpr int WARPS_N    = 2;
static constexpr int WARP_M     = BLOCK_M / WARPS_M;   // 32
static constexpr int WARP_N     = BLOCK_N / WARPS_N;   // 32
static constexpr int TILES_M    = WARP_M / 16;         // 2
static constexpr int TILES_N    = WARP_N / 8;          // 4
static constexpr int NUM_STAGES = 2;
static constexpr int SMEM_STRIDE = BLOCK_K / 2;        // 32 bytes / row

static constexpr int TILE_A_BYTES  = BLOCK_M * SMEM_STRIDE;   // 2048
static constexpr int TILE_B_BYTES  = BLOCK_N * SMEM_STRIDE;   // 2048
static constexpr int TILE_SA_BYTES = BLOCK_M * 2;             // 128 (fp16)
static constexpr int TILE_SB_BYTES = BLOCK_N * 2;             // 128 (fp16)
static constexpr int STAGE_BYTES   = TILE_A_BYTES + TILE_B_BYTES
                                   + TILE_SA_BYTES + TILE_SB_BYTES;  // 4352
static constexpr int NUM_THREADS   = WARP_SZ * NUM_WARPS;     // 128

__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
    using FragA = typename CutlassMmaOp::FragmentA;
    using FragB = typename CutlassMmaOp::FragmentB;
    using FragC = typename CutlassMmaOp::FragmentC;
    static_assert(sizeof(FragA) == sizeof(uint4),   "FragA size mismatch");
    static_assert(sizeof(FragB) == sizeof(uint2),   "FragB size mismatch");
    static_assert(sizeof(FragC) == sizeof(int) * 4, "FragC size mismatch");
    FragA &fa = reinterpret_cast<FragA&>(a);
    FragB &fb = reinterpret_cast<FragB&>(b);
    FragC &fc = reinterpret_cast<FragC&>(c);
    CutlassMmaOp mma;
    mma(fc, fa, fb, fc);
}

using SmemLayoutAB = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
    cutlass::sizeof_bits<cutlass::int4b_t>::value, BLOCK_K>;

__device__ __forceinline__ int cutlass_smem_byte_off(int row, int col_byte) {
    SmemLayoutAB layout = SmemLayoutAB::packed({BLOCK_M, BLOCK_K});
    int col_int4 = col_byte * 2;
    int off_int4 = layout({row, col_int4});
    return off_int4 / 2;
}

__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait(int n) {
    if (n <= 0)      asm volatile("cp.async.wait_group 0;\n");
    else if (n == 1) asm volatile("cp.async.wait_group 1;\n");
    else             asm volatile("cp.async.wait_group 2;\n");
}

__device__ __forceinline__ uint4 load_a_frag(const uint8_t *smem_base, int abs_row_start) {
    int lane      = threadIdx.x & (WARP_SZ - 1);
    int local_row = lane & 15;
    int col_byte  = (lane >> 4) * 16;
    int abs_row   = abs_row_start + local_row;
    int smem_off  = cutlass_smem_byte_off(abs_row, col_byte);
    uint32_t sp   = __cvta_generic_to_shared(smem_base + smem_off);
    uint4 a;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
        : "r"(sp));
    return a;
}

__device__ __forceinline__ uint2 load_b_frag(const uint8_t *smem_base, int abs_row_start) {
    int lane      = threadIdx.x & (WARP_SZ - 1);
    int local_row = lane & 7;
    int col_byte  = ((lane >> 3) & 1) * 16;
    int abs_row   = abs_row_start + local_row;
    int smem_off  = cutlass_smem_byte_off(abs_row, col_byte);
    uint32_t sp   = __cvta_generic_to_shared(smem_base + smem_off);
    uint2 b;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b.x), "=r"(b.y)
        : "r"(sp));
    return b;
}

// ============================================================================
// __launch_bounds__(128, 6): 128 threads/block x 6 blocks/SM = 24 active
// warps/SM. Budget is 65536/(6*128) = 85 regs/thread, which comfortably fits
// a ~60-reg design but guards against the compiler silently allowing 100+ reg
// usage (which would drop us to 4 blocks/SM and match the old 16-warp cliff).
__global__ __launch_bounds__(NUM_THREADS, 6) void gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,
    const half    *__restrict__ scales_B,
    half          *__restrict__ C,
    int M, int N, int K, int group_size)
{
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid    = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    const int warp_m_id = warpId / WARPS_N;
    const int warp_n_id = warpId % WARPS_N;
    const int warp_bm   = warp_m_id * WARP_M;
    const int warp_bn   = warp_n_id * WARP_N;

    extern __shared__ uint8_t smem[];
    uint8_t *sA[NUM_STAGES];
    uint8_t *sB[NUM_STAGES];
    half    *sSa[NUM_STAGES];
    half    *sSb[NUM_STAGES];
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; i++) {
        uint8_t *base = smem + i * STAGE_BYTES;
        sA[i]  = base;
        sB[i]  = base + TILE_A_BYTES;
        sSa[i] = reinterpret_cast<half*>(base + TILE_A_BYTES + TILE_B_BYTES);
        sSb[i] = reinterpret_cast<half*>(base + TILE_A_BYTES + TILE_B_BYTES + TILE_SA_BYTES);
    }

    float acc[TILES_M][TILES_N][4];
    #pragma unroll
    for (int tm = 0; tm < TILES_M; tm++)
    #pragma unroll
        for (int tn = 0; tn < TILES_N; tn++)
    #pragma unroll
            for (int i = 0; i < 4; i++)
                acc[tm][tn][i] = 0.f;

    // ---- Cooperative tile loader ----
    // A tile = 64 rows x 32 bytes = 2048 B -> 128 threads x 16 B each.
    // B tile = same. Each of the 128 threads issues one cp.async for A and one
    // for B (total 256 cp.async / block / K-iter for weights).
    // Scales = 64 halves each = 128 B. 8 threads cover scales_A (16 B each),
    // the next 8 cover scales_B.
    const size_t tile_bytes = (size_t)(BLOCK_M * (BLOCK_K / 2));   // 2048
    const size_t a_block_base = (size_t)blockIdx.y * (size_t)num_k_tiles * tile_bytes;
    const size_t b_block_base = (size_t)blockIdx.x * (size_t)num_k_tiles * tile_bytes;

    auto load_tile = [&](int kt, int s) {
        int row      = tid / 2;
        int half_idx = tid % 2;
        int col_byte = half_idx * 16;
        int smem_off = cutlass_smem_byte_off(row, col_byte);
        size_t tile_off = (size_t)kt * tile_bytes
                        + (size_t)row * (BLOCK_K / 2) + half_idx * 16;
        cp_async_16(sA[s] + smem_off, A + a_block_base + tile_off, true);
        cp_async_16(sB[s] + smem_off, B + b_block_base + tile_off, true);

        if (tid < 8) {
            size_t sa_off = (size_t)blockIdx.y * num_groups * BLOCK_M
                          + (size_t)kt * BLOCK_M
                          + (size_t)tid * 8;
            cp_async_16(
                reinterpret_cast<uint8_t*>(sSa[s]) + tid * 16,
                scales_A + sa_off,
                true);
        } else if (tid < 16) {
            int idx = tid - 8;
            size_t sb_off = (size_t)blockIdx.x * num_groups * BLOCK_N
                          + (size_t)kt * BLOCK_N
                          + (size_t)idx * 8;
            cp_async_16(
                reinterpret_cast<uint8_t*>(sSb[s]) + idx * 16,
                scales_B + sb_off,
                true);
        }
        cp_commit();
    };

    #pragma unroll
    for (int i = 0; i < NUM_STAGES - 1; i++) {
        if (i < num_k_tiles) load_tile(i, i);
    }

    // ==== Main K-loop ====
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt % NUM_STAGES;
        int prefetch_kt = kt + NUM_STAGES - 1;
        if (prefetch_kt < num_k_tiles) {
            load_tile(prefetch_kt, prefetch_kt % NUM_STAGES);
        }
        int max_pending = num_k_tiles - 1 - kt;
        if (max_pending > NUM_STAGES - 1) max_pending = NUM_STAGES - 1;
        cp_wait(max_pending);
        __syncthreads();

        // Per-K-iter register-cached scales_A (quad-broadcast from SMEM).
        float sa_lo[TILES_M], sa_hi[TILES_M];
        #pragma unroll
        for (int tm = 0; tm < TILES_M; tm++) {
            float v_lo = 0.f, v_hi = 0.f;
            if ((laneId & 3) == 0) {
                int row_lo = warp_bm + tm * 16 + laneId / 4;
                int row_hi = row_lo + 8;
                v_lo = __half2float(sSa[s][row_lo]);
                v_hi = __half2float(sSa[s][row_hi]);
            }
            sa_lo[tm] = __shfl_sync(0xffffffff, v_lo, laneId & ~3);
            sa_hi[tm] = __shfl_sync(0xffffffff, v_hi, laneId & ~3);
        }

        uint4 af[TILES_M];
        #pragma unroll
        for (int tm = 0; tm < TILES_M; tm++) {
            af[tm] = load_a_frag(sA[s], warp_bm + tm * 16);
        }

        // 8 MMAs per warp per K-iter (2 TILES_M x 4 TILES_N).
        #pragma unroll
        for (int tn = 0; tn < TILES_N; tn++) {
            int frag_bn = warp_bn + tn * 8;
            uint2 bf = load_b_frag(sB[s], frag_bn);
            float v_sb0 = 0.f, v_sb1 = 0.f;
            if (laneId < 4) {
                int bc0 = frag_bn + laneId * 2;
                v_sb0 = __half2float(sSb[s][bc0]);
                v_sb1 = __half2float(sSb[s][bc0 + 1]);
            }
            float sb0 = __shfl_sync(0xffffffff, v_sb0, laneId & 3);
            float sb1 = __shfl_sync(0xffffffff, v_sb1, laneId & 3);
            #pragma unroll
            for (int tm = 0; tm < TILES_M; tm++) {
                float sa_lo_sb0 = sa_lo[tm] * sb0;
                float sa_lo_sb1 = sa_lo[tm] * sb1;
                float sa_hi_sb0 = sa_hi[tm] * sb0;
                float sa_hi_sb1 = sa_hi[tm] * sb1;
                int p[4] = {0,0,0,0};
                mma_s4(af[tm], bf, p);
                acc[tm][tn][0] = __fmaf_rn((float)p[0], sa_lo_sb0, acc[tm][tn][0]);
                acc[tm][tn][1] = __fmaf_rn((float)p[1], sa_lo_sb1, acc[tm][tn][1]);
                acc[tm][tn][2] = __fmaf_rn((float)p[2], sa_hi_sb0, acc[tm][tn][2]);
                acc[tm][tn][3] = __fmaf_rn((float)p[3], sa_hi_sb1, acc[tm][tn][3]);
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: vectorized half2 stores ----
    #pragma unroll
    for (int tm = 0; tm < TILES_M; tm++) {
        int m_lo = bm + warp_bm + tm * 16 + laneId / 4;
        int m_hi = m_lo + 8;
        #pragma unroll
        for (int tn = 0; tn < TILES_N; tn++) {
            int col0 = bn + warp_bn + tn * 8 + (laneId % 4) * 2;
            half2 lo_pair = __floats2half2_rn(acc[tm][tn][0], acc[tm][tn][1]);
            half2 hi_pair = __floats2half2_rn(acc[tm][tn][2], acc[tm][tn][3]);
            if (m_lo < M) {
                *reinterpret_cast<half2*>(&C[m_lo * N + col0]) = lo_pair;
            }
            if (m_hi < M) {
                *reinterpret_cast<half2*>(&C[m_hi * N + col0]) = hi_pair;
            }
        }
    }
}

torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed, torch::Tensor B_packed,
    torch::Tensor scales_A, torch::Tensor scales_B, int group_size)
{
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda());
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8);
    int M = A_packed.size(0), K = A_packed.size(1) * 2, N = B_packed.size(0);

    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);
    int smem = NUM_STAGES * STAGE_BYTES;

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
