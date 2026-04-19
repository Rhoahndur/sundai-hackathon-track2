#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// CUTLASS: only used for its RowMajorTensorOpMultiplicandCrosswise SMEM layout
// (gives us the canonical ldmatrix-friendly swizzle for INT4) and as a typed
// wrapper around the sm_80 INT4 MMA PTX. Build requires v4.4.2+ headers on
// NVCC_APPEND_FLAGS="-I/path/to/cutlass/include".
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/arch/mma.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/layout/tensor_op_multiplicand_sm80.h>

// ---- MMA wrapper: m16n8k64 INT4xINT4 -> INT32 ----
// CUTLASS's Mma specialization pins to OpMultiplyAddSaturate; identical to
// OpMultiplyAdd for our data (max |acc| = 64*7*8 = 3584, well inside int32).
using CutlassMmaOp = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<16, 8, 64>,
    32,
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int,              cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAddSaturate
>;

// ============================================================================
// Offline-quant kernel: writes block-major A / scales_A for this GEMM.
// ============================================================================
__global__ void quantize_int4_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int M,
    int K,
    int group_size)
{
    constexpr int QUANT_BLOCK_M = 128;

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
// GEMM kernel -- COMET-style W4A4 pipeline (v2: live-state trimmed)
//
// The architecture attacks the structural ceiling the old 200-TOPs kernel hit:
// three __syncthreads per K-iter, driven by a smem_sb staging step that was
// separate from the weights cp.async group.
//
// Key changes vs the old kernel:
//   - scales_A and scales_B live in the SAME cp.async group as the weight
//     tiles. A single committed group drains A + B + scales_A + scales_B
//     together, so the "gmem -> fp32 smem_sb + extra sync" stage is gone.
//   - Down to 2 __syncthreads per K-iter (top-of-iter after cp_wait +
//     end-of-iter to guard the prefetch stage-reuse race).
//   - Single ldmatrix.x2 B-frag load. The paired x4 variant regressed on
//     Ampere due to the extra register live range, and the v1 sb_pair
//     register cache regressed 30% because it pushed the kernel out of the
//     2-block/SM tier. v2 deliberately reads scales_B per-tn from SMEM and
//     shfl-broadcasts inside the tile loop -- keeps the live-state flat.
//   - __launch_bounds__(256, 2) pins the register budget to 128 regs/thread
//     so the compiler can't silently drop occupancy.
//
// Fixed layout:
//   BLOCK_M=BLOCK_N=128, BLOCK_K=group_size=64, 8 warps in a 4x2 grid.
//   Per stage SMEM:
//     A  : 128 rows x 32 bytes = 4096 B (CUTLASS crosswise swizzle)
//     B  : 128 rows x 32 bytes = 4096 B (CUTLASS crosswise swizzle)
//     sA : 128 halves           =  256 B (linear, no swizzle)
//     sB : 128 halves           =  256 B (linear, no swizzle)
//   2 stages -> 17.4 KB / block, still comfortably inside the 5-block/SM tier
//   on sm_86 (96 KB usable L1/SMEM per SM).
// ============================================================================

static constexpr int BLOCK_M    = 128;
static constexpr int BLOCK_N    = 128;
static constexpr int BLOCK_K    = 64;      // must equal quant group_size
static constexpr int WARP_SZ    = 32;
static constexpr int NUM_WARPS  = 8;
static constexpr int WARPS_M    = 4;
static constexpr int WARPS_N    = 2;
static constexpr int WARP_M     = BLOCK_M / WARPS_M;   // 32
static constexpr int WARP_N     = BLOCK_N / WARPS_N;   // 64
static constexpr int TILES_M    = WARP_M / 16;         // 2
static constexpr int TILES_N    = WARP_N / 8;          // 8
static constexpr int NUM_STAGES = 2;
static constexpr int SMEM_STRIDE = BLOCK_K / 2;        // 32 bytes / row

// Per-stage SMEM byte layout.
static constexpr int TILE_A_BYTES  = BLOCK_M * SMEM_STRIDE;   // 4096
static constexpr int TILE_B_BYTES  = BLOCK_N * SMEM_STRIDE;   // 4096
static constexpr int TILE_SA_BYTES = BLOCK_M * 2;             // 256 (fp16)
static constexpr int TILE_SB_BYTES = BLOCK_N * 2;             // 256 (fp16)
static constexpr int STAGE_BYTES   = TILE_A_BYTES + TILE_B_BYTES
                                   + TILE_SA_BYTES + TILE_SB_BYTES;

// ---- MMA wrapper routed through CUTLASS's typed Mma ----
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

// ---- CUTLASS-layout-driven SMEM offset ----
// Swizzle shared by A and B (same int4 crosswise pattern).
using SmemLayoutAB = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
    cutlass::sizeof_bits<cutlass::int4b_t>::value, BLOCK_K>;

__device__ __forceinline__ int cutlass_smem_byte_off(int row, int col_byte) {
    SmemLayoutAB layout = SmemLayoutAB::packed({BLOCK_M, BLOCK_K});
    int col_int4 = col_byte * 2;   // 2 int4 per byte
    int off_int4 = layout({row, col_int4});
    return off_int4 / 2;
}

// ---- cp.async helpers ----
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

// ---- Frag loads ----
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
// __launch_bounds__(256, 2) pins the per-block register budget to
// 65536/(2*256) = 128 regs/thread max. This prevents the compiler from
// over-allocating and dropping to 1 block/SM (occupancy cliff) in the v1
// experiment and forces the occupancy tier we designed the SMEM layout for.
__global__ __launch_bounds__(256, 2) void gemm_int4_kernel(
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

    // ---- SMEM pointers (A, B, scales_A, scales_B for each stage) ----
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

    // FP32 accumulators: one m16n8k64 MMA's 4-element output per (tm, tn).
    float acc[TILES_M][TILES_N][4];
    #pragma unroll
    for (int tm = 0; tm < TILES_M; tm++)
    #pragma unroll
        for (int tn = 0; tn < TILES_N; tn++)
    #pragma unroll
            for (int i = 0; i < 4; i++)
                acc[tm][tn][i] = 0.f;

    // ---- Cooperative tile loader (weights + scales, one cp.async group) ----
    //   - 256 threads x 16 B = 4096 B per weight tile -> each thread issues one
    //     cp.async_16 for A and one for B.
    //   - First 16 threads also issue 16 B of scales_A (16 * 8 halves = 128).
    //   - Next 16 threads issue 16 B of scales_B the same way.
    // The writes to the scales SMEM do not swizzle: the scales are read by
    // index, not by ldmatrix.
    const size_t tile_bytes = (size_t)(BLOCK_M * (BLOCK_K / 2));   // 4096
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

        // Scales live in the same cp.async group as the weights they pair
        // with. Group index == kt (BLOCK_K == group_size).
        if (tid < 16) {
            size_t sa_off = (size_t)blockIdx.y * num_groups * BLOCK_M
                          + (size_t)kt * BLOCK_M
                          + (size_t)tid * 8;
            cp_async_16(
                reinterpret_cast<uint8_t*>(sSa[s]) + tid * 16,
                scales_A + sa_off,
                true);
        } else if (tid < 32) {
            int idx = tid - 16;
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

    // Pipeline warm-up.
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
        __syncthreads();   // Single top-of-iter barrier -- weights + scales
                           // all visible from this point.

        // -------- Per-K-iter register-cached scales --------
        // scales_A: quad-broadcast one fp16 per row into sa_lo/sa_hi.
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

        // -------- Preload A-frags (reused across TILES_N B-frags) --------
        uint4 af[TILES_M];
        #pragma unroll
        for (int tm = 0; tm < TILES_M; tm++) {
            af[tm] = load_a_frag(sA[s], warp_bm + tm * 16);
        }

        // -------- MMA mainloop (16 MMAs per warp, single ldmatrix.x2 B) --------
        // scales_B values are shfl-broadcast per-tn from SMEM directly rather
        // than hoisted into a register cache. The v1 pre-broadcast pattern
        // (sb_pair[TILES_N][2]) added +16 fp32 regs/thread and dropped the
        // kernel out of the 2-block/SM tier -- net -30% on the benchmark.
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
        __syncthreads();   // End-of-iter: guard against next prefetch race.
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

// ---- Host wrapper (drop-in replacement) ----
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
    dim3 block(WARP_SZ * NUM_WARPS);
    int smem = NUM_STAGES * STAGE_BYTES;

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
