"""Vulkan backend: fused kernels through torchvulkan.

torchvulkan (https://github.com/zainsharief/torchvulkan) runs PyTorch on Vulkan devices as
``device="vulkan"``. Recent versions provide ops that cover two hot paths:

- ``int8_linear``: ``torch.ops.torchvulkan.quantize_int8_rowwise(x)`` quantizes the activations
  per row in one kernel, and ``torch.ops.torchvulkan.int8_scaled_mm(a, b, a_scale, b_scale, bias,
  out_dtype)`` runs the INT8 matmul with the rescale to ``out_dtype`` (and the bias) applied as the
  INT32 results leave the matrix units, so they never go through device memory. Input activations,
  ConvRot and the residual form use the eager helpers.
- ``rms_rope`` and its variants: ``torch.ops.torchvulkan.rms_norm_rope`` normalizes each head and
  applies the rotary embedding in one kernel. Partial rotary dims and other ``freqs_cis`` layouts
  go to the eager backend.

They follow the eager backend's order of operations. With a torchvulkan that lacks an op, its
calls fall through to the eager backend.
"""

from __future__ import annotations

import functools
import importlib.util

import torch

from comfy_kitchen.backends._activations import apply_input_act as _apply_input_act
from comfy_kitchen.backends._activations import apply_residual as _apply_residual
from comfy_kitchen._rope_utils import trim_rope_freqs
from comfy_kitchen.constraints import ExactDims, FunctionConstraints, ParamConstraint, ValidationResult
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

__all__ = ["int8_linear", "rms_rope", "rms_rope1", "rms_rope_split_half", "rms_rope_split_half1"]


def _torchvulkan_ops(*names: str):
    """The torchvulkan op namespace, or None when it lacks any of the named ops."""
    try:
        import torchvulkan  # noqa: F401  registers the ops
    except ImportError:
        return None
    ops = torch.ops.torchvulkan
    return ops if all(hasattr(ops, name) for name in names) else None


_INT8_OPS = ("int8_scaled_mm", "quantize_int8_rowwise")


def _validate_int8_linear(kwargs) -> ValidationResult:
    if _torchvulkan_ops(*_INT8_OPS) is None:
        return ValidationResult.fail("x", "this torchvulkan has no fused int8 ops")
    weight = kwargs.get("weight")
    if isinstance(weight, torch.Tensor) and weight.dim() != 2:
        return ValidationResult.fail("weight", "expected a 2D [N, K] weight")
    weight_scale = kwargs.get("weight_scale")
    if isinstance(weight, torch.Tensor) and isinstance(weight_scale, torch.Tensor):
        if weight_scale.numel() not in (1, weight.shape[0]):
            return ValidationResult.fail("weight_scale", "must be scalar or per output channel")
    if kwargs.get("residual") is not None and kwargs.get("residual_scale") is None:
        return ValidationResult.fail("residual_scale", "required when residual is provided")
    return ValidationResult.ok()


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 linear with dynamic per-row activation quantization, as the eager backend's."""
    ops = _torchvulkan_ops(*_INT8_OPS)
    x = _apply_input_act(x, input_act, input_act_weight, input_act_eps)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )
    weight = weight.to(device=x.device)

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_8, x_scale = ops.quantize_int8_rowwise(x.reshape(-1, x.shape[-1]))
    # weight.T is the [K, N] operand as a view; the kernel reads it without a copy
    result = ops.int8_scaled_mm(x_8, weight.T, x_scale, weight_scale, bias, out_dtype)
    result = result.reshape(*orig_shape[:-1], weight.shape[0])
    return _apply_residual(result, residual, residual_scale)


# RMS norm + RoPE (qk-norm and rotary embedding of attention inputs) in one torchvulkan kernel


def _rope_freqs(x: torch.Tensor, freqs_cis: torch.Tensor, split_half: bool) -> torch.Tensor | None:
    """freqs_cis as the kernel takes it ([., ., ., D/2, 2, 2] broadcast over x), or None."""
    if not split_half:  # the eager interleaved rope trims extra positions; its split-half one does not
        freqs_cis = trim_rope_freqs(x, freqs_cis)
    if freqs_cis.ndim != 6 or tuple(freqs_cis.shape[3:]) != (x.shape[-1] // 2, 2, 2):
        return None
    if any(f not in (1, s) for f, s in zip(freqs_cis.shape[:3], x.shape[:3], strict=True)):
        return None
    return freqs_cis


def _validate_rms_rope(kwargs, *, split_half: bool) -> ValidationResult:
    if _torchvulkan_ops("rms_norm_rope") is None:
        return ValidationResult.fail("x", "this torchvulkan has no rms_norm_rope")
    if kwargs.get("rot_dim"):
        tensors = [kwargs.get(n) for n in ("q", "x")]
        dim = next((t.shape[-1] for t in tensors if isinstance(t, torch.Tensor)), None)
        if dim is not None and kwargs["rot_dim"] != dim:
            return ValidationResult.fail("rot_dim", "partial rotary dims are not fused")
    freqs = kwargs.get("freqs_cis")
    for name in ("q", "k", "x"):
        x = kwargs.get(name)
        if not isinstance(x, torch.Tensor):
            continue
        if x.ndim != 4 or x.shape[-1] % 2:
            return ValidationResult.fail(name, "expected a 4D input with an even last dimension")
        if isinstance(freqs, torch.Tensor) and _rope_freqs(x, freqs, split_half) is None:
            return ValidationResult.fail("freqs_cis", "layout not supported by the fused kernel")
    return ValidationResult.ok()


def _rms_rope_one(x, freqs_cis, scale, epsilon, split_half):
    ops = _torchvulkan_ops("rms_norm_rope")
    return ops.rms_norm_rope(x, scale, _rope_freqs(x, freqs_cis, split_half), epsilon, split_half)


def rms_rope1(x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """rms_norm(x, weight=scale) followed by the interleaved rotary embedding, in one kernel."""
    return _rms_rope_one(x, freqs_cis, scale, epsilon, split_half=False)


def rms_rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6):
    if k_scale is None:
        k_scale = q_scale
    return (_rms_rope_one(q, freqs_cis, q_scale, epsilon, False),
            _rms_rope_one(k, freqs_cis, k_scale, epsilon, False))


def rms_rope_split_half1(x, freqs_cis, scale, epsilon: float = 1e-6):
    return _rms_rope_one(x, freqs_cis, scale, epsilon, split_half=True)


def rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6, rot_dim: int = 0):
    if k_scale is None:
        k_scale = q_scale
    return (_rms_rope_one(q, freqs_cis, q_scale, epsilon, True),
            _rms_rope_one(k, freqs_cis, k_scale, epsilon, True))


def _build_constraints() -> dict[str, FunctionConstraints]:
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    constraints = {
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                "weight_scale": ParamConstraint(dtypes=floats),
                "bias": ParamConstraint(dtypes=floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=frozenset({"vulkan"}),
            call_rules=(_validate_int8_linear,),
        ),
    }
    four_d = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),))
    scale = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),))
    freqs = ParamConstraint(dtypes=floats)
    pair = {"q": four_d, "k": four_d, "freqs_cis": freqs, "q_scale": scale, "k_scale": scale}
    one = {"x": four_d, "freqs_cis": freqs, "scale": scale}
    for name, params, split_half in (("rms_rope", pair, False), ("rms_rope_split_half", pair, True),
                                     ("rms_rope1", one, False), ("rms_rope_split_half1", one, True)):
        constraints[name] = FunctionConstraints(
            params=params, default_devices=frozenset({"vulkan"}),
            call_rules=(functools.partial(_validate_rms_rope, split_half=split_half),))
    return constraints


# Registered whenever torchvulkan is installed, without importing it (it initialises Vulkan); the
# ops are looked up on the first call, and the constraints keep every other device on its own
# backend.
if importlib.util.find_spec("torchvulkan") is not None:
    registry.register(
        name="vulkan",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )
else:
    registry.mark_unavailable("vulkan", "torchvulkan is not installed")
