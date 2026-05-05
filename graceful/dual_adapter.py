"""
Coupled Manifold — DualAdapter module.
LoRA + Anti-LoRA dual adapter, MLX native.
"""
import mlx.core as mx
import mlx.nn as nn

from .config import RANK


class DualAdapter(nn.Module):
    """LoRA + Anti-LoRA dual adapter — MLX native implementation."""
    def __init__(self, base, rank: int = RANK):
        super().__init__()
        self.base = base
        # Infer dimensions from base layer (handles Linear and QuantizedLinear)
        if hasattr(base, 'scales'):                      # QuantizedLinear (4-bit)
            out_f = base.scales.shape[0]
            in_f  = base.scales.shape[1] * base.group_size
        elif hasattr(base, 'weight'):                    # Regular Linear
            out_f, in_f = base.weight.shape[0], base.weight.shape[1]
        else:
            raise ValueError(f"DualAdapter: unrecognised layer type {type(base)}")
        self.lA = mx.random.normal([rank, in_f])  * 0.01
        self.lB = mx.zeros([out_f, rank])
        self.aA = mx.random.normal([rank, in_f])  * 0.01
        self.aB = mx.zeros([out_f, rank])
        self.scale   = 2.0
        self.a_str   = 0.1
        self.lora_on = True
        self.anti_on = False

    def __call__(self, x):
        out = self.base(x)
        if self.lora_on:
            out = out + (x @ self.lA.T) @ self.lB.T * self.scale
        if self.anti_on:
            out = out + (x @ self.aA.T) @ self.aB.T * self.a_str
        return out
