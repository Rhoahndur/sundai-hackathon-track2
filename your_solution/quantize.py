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


def quantize_weights(weight: torch.Tensor, group_size: int = 64) -> dict:
    """Quantize a FP16 weight tensor to packed INT4 format.

    Args:
        weight: [N, K] float16 weight tensor.
        group_size: Number of elements per quantization group.

    Returns:
        dict with:
            "weight_packed": [N, K//2] uint8 tensor (packed INT4)
            "weight_scales": [N, K//group_size] float16 tensor (per-group scales)
            "group_size": int
    """
    assert weight.dim() == 2, "weight must be 2D [N, K]"
    N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 2 == 0, "group_size must be even"

    num_groups = K // group_size

    # Work in float32 for precision
    w = weight.float().reshape(N, num_groups, group_size)

    # Per-group symmetric scale: scale = max(|x|) / 7
    max_abs = w.abs().amax(dim=-1, keepdim=True)  # [N, num_groups, 1]
    scale = max_abs / 7.0
    rscale = torch.where(max_abs > 0, 7.0 / max_abs, torch.zeros_like(max_abs))

    # Quantize: round to nearest, clamp to [-8, 7]
    q = (w * rscale).round().clamp(-8, 7).to(torch.int8)  # [N, num_groups, group_size]
    q = q.reshape(N, K)

    # Pack two INT4 values per byte: low nibble = even, high nibble = odd.
    # Cast to uint8 *before* the shift so that high nibbles (e.g. 15 for -1)
    # produce 240 (0xf0) without intermediate signed-int8 wrap.
    even = (q[:, 0::2].to(torch.uint8) & 0x0F)
    odd = ((q[:, 1::2].to(torch.uint8) & 0x0F) << 4)
    packed = odd | even  # [N, K//2]

    scales = scale.squeeze(-1).half()  # [N, num_groups]

    return {
        "weight_packed": packed,
        "weight_scales": scales,
        "group_size": group_size,
    }
