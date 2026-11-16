"""Vulkan backend: fused INT8 linear through torchvulkan.

torchvulkan (https://github.com/zainsharief/torchvulkan) runs PyTorch on Vulkan devices as
``device="vulkan"``. Recent versions provide two ops that cover ``int8_linear``'s hot path:

- ``torch.ops.torchvulkan.quantize_int8_rowwise(x)``: the per-row INT8 quantization of the
  activations in one kernel;
- ``torch.ops.torchvulkan.int8_scaled_mm(a, b, a_scale, b_scale, bias, out_dtype)``: the INT8
  matmul with the rescale to ``out_dtype`` (and the bias) applied as the INT32 results leave the
  matrix units, so they never go through device memory.

Both follow the eager backend's order of operations and give the same results. Everything else
(input activations, ConvRot, the residual form) uses the eager helpers. With an older torchvulkan
that lacks the ops, calls fall through to the eager backend.
"""

from __future__ import annotations

import importlib.util

import torch

from comfy_kitchen.backends._activations import apply_input_act as _apply_input_act
from comfy_kitchen.backends._activations import apply_residual as _apply_residual
from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint, ValidationResult
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

__all__ = ["int8_linear"]


def _torchvulkan_ops():
    """The torchvulkan op namespace, or None when the fused INT8 ops are not available."""
    try:
        import torchvulkan  # noqa: F401  registers the ops
    except ImportError:
        return None
    ops = torch.ops.torchvulkan
    if not (hasattr(ops, "int8_scaled_mm") and hasattr(ops, "quantize_int8_rowwise")):
        return None
    return ops


def _validate_int8_linear(kwargs) -> ValidationResult:
    if _torchvulkan_ops() is None:
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
    ops = _torchvulkan_ops()
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


def _build_constraints() -> dict[str, FunctionConstraints]:
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    return {
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
