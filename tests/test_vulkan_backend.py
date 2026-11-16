"""The Vulkan backend's int8_linear (torchvulkan's fused ops) against the eager backend.

Runs only with torchvulkan installed, a Vulkan device, and a torchvulkan that has the fused ops.
The eager reference runs on the CPU, where every op rounds exactly, so the results must match bit
for bit.
"""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear
from comfy_kitchen.registry import registry

torchvulkan = pytest.importorskip("torchvulkan")
if not torchvulkan.is_available():
    pytest.skip("no Vulkan device", allow_module_level=True)
if not hasattr(torch.ops.torchvulkan, "int8_scaled_mm"):
    pytest.skip("this torchvulkan has no fused int8 ops", allow_module_level=True)

DEVICE = "vulkan"


def _dtypes():
    dtypes = [torch.float32, torch.float16]
    if torchvulkan.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def test_vulkan_backend_is_selected():
    x = torch.randn(4, 64, device=DEVICE)
    w = torch.randint(-128, 128, (32, 64), dtype=torch.int8, device=DEVICE)
    kwargs = {"x": x, "weight": w, "weight_scale": torch.ones(1, device=DEVICE), "bias": None,
              "out_dtype": torch.float32, "convrot": False, "convrot_groupsize": 256}
    assert registry.get_capable_backend("int8_linear", kwargs) == "vulkan"


def test_other_devices_keep_their_backend():
    x = torch.randn(4, 64)
    w = torch.randint(-128, 128, (32, 64), dtype=torch.int8)
    kwargs = {"x": x, "weight": w, "weight_scale": torch.ones(1), "bias": None,
              "out_dtype": torch.float32, "convrot": False, "convrot_groupsize": 256}
    assert registry.get_capable_backend("int8_linear", kwargs) != "vulkan"


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("shape", [(1, 256, 64), (2, 77, 384, 300), (130, 512, 70)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("convrot", [False, True])
def test_int8_linear_matches_eager(dtype, shape, with_bias, per_channel, convrot):
    *lead, k, n = shape
    g = torch.Generator().manual_seed(k * 7 + n)
    x = torch.randn(*lead, k, generator=g).to(dtype)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=g)
    w_scale = torch.rand(n if per_channel else 1, generator=g) * 0.01 + 1e-4
    bias = torch.randn(n, generator=g).to(dtype) if with_bias else None

    expected = eager_int8_linear(x, w, w_scale, bias, dtype, convrot=convrot, convrot_groupsize=64)
    result = ck.int8_linear(x.to(DEVICE), w.to(DEVICE), w_scale.to(DEVICE),
                            None if bias is None else bias.to(DEVICE), dtype,
                            convrot=convrot, convrot_groupsize=64)
    assert result.shape == expected.shape and result.dtype == dtype
    if convrot:
        # the rotation is a float matmul, whose summation order differs between devices, so the
        # quantized activations can differ by one step near rounding boundaries
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-2, atol=2e-2)
    else:
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
