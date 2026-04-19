"""Offline weight quantization -- YOUR SOLUTION.

Modify this file to implement your own quantization strategy.
The standard implementation uses round-to-nearest symmetric INT4
with group_size=64. You may change the algorithm or the group_size, as long as:
  1. The function signature stays the same.
  2. The output format is compatible with your CUDA gemm_int4 kernel.
  3. The end-to-end result passes the cosine similarity threshold (>0.98).

The packed format convention:
  - Two signed INT4 values per uint8 byte
  - Low nibble = even element, high nibble = odd element
  - Scales are FP16, one per group
"""

import torch


BLOCK_N = 64  # must match kernel.cu BLOCK_N (v3: 64x64 tile)


def quantize_weights(weight: torch.Tensor, group_size: int = 64) -> dict:
    """Quantize a FP16 weight tensor to packed INT4 with block-major reordering.

    The *values* are standard symmetric-per-group INT4 (scale = max|x|/7,
    round-to-nearest, clamp to [-8, 7]). The *layout* is reordered to match
    what the CUDA GEMM kernel wants to see from cp.async:

        weight_packed[n_block, k_block, row_in_block, byte_in_row]
            n_block      = row // BLOCK_N                      (outer)
            k_block      = k_byte // (group_size / 2)           (second)
            row_in_block = row %  BLOCK_N
            byte_in_row  = k_byte % (group_size / 2)

    Each (n_block, k_block) is one contiguous 4096-byte chunk (128 rows x
    32 bytes), so a 256-thread cp.async fans out across one contiguous
    region instead of 256 stride-K/2 rows. Same repack for weight_scales:
    [n_block, k_block, row_in_block].

    The tensor shapes exposed to the benchmark are still [N, K/2] and
    [N, num_groups]; only the byte order changes. The kernel reads by
    flat offset so it doesn't care about the 2D interpretation.

    Args:
        weight: [N, K] float16 weight tensor.
        group_size: Number of elements per quantization group (must == kernel BLOCK_K).

    Returns:
        dict with:
            "weight_packed": [N, K//2] uint8 tensor, block-major byte order
            "weight_scales": [N, K//group_size] float16, block-major order
            "group_size": int
    """
    assert weight.dim() == 2, "weight must be 2D [N, K]"
    N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 2 == 0, "group_size must be even"
    assert N % BLOCK_N == 0, f"N ({N}) must be divisible by BLOCK_N ({BLOCK_N})"

    num_groups = K // group_size
    num_n_blocks = N // BLOCK_N
    num_k_blocks = num_groups  # BLOCK_K == group_size, so k_blocks == groups

    # --- Standard INT4 quant (unchanged values) ---
    w = weight.float().reshape(N, num_groups, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 7.0
    rscale = torch.where(max_abs > 0, 7.0 / max_abs, torch.zeros_like(max_abs))
    q = (w * rscale).round().clamp(-8, 7).to(torch.int8)
    q = q.reshape(N, K)

    even = (q[:, 0::2] & 0xF).to(torch.uint8)
    odd = ((q[:, 1::2] & 0xF) << 4).to(torch.uint8)
    packed = odd | even  # [N, K//2]

    # --- Block-major reorder ---
    # packed currently has axis meaning (row, k_byte). Reshape to
    # (n_block, row_in_block, k_block, byte_in_row), permute so (n_block,
    # k_block, row_in_block, byte_in_row) lays out contiguously, then
    # flatten back to [N, K/2].
    packed = packed.reshape(num_n_blocks, BLOCK_N, num_k_blocks, group_size // 2)
    packed = packed.permute(0, 2, 1, 3).contiguous()
    weight_packed = packed.reshape(N, K // 2)

    # Scales: (row, group) -> (n_block, k_block, row_in_block) contiguous.
    scales = scale.squeeze(-1).half()  # [N, num_groups]
    scales = scales.reshape(num_n_blocks, BLOCK_N, num_k_blocks)
    scales = scales.permute(0, 2, 1).contiguous()
    weight_scales = scales.reshape(N, num_groups)

    return {
        "weight_packed": weight_packed,
        "weight_scales": weight_scales,
        "group_size": group_size,
    }
