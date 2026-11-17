"""The Vulkan backend (torchvulkan's fused ops) against the eager backend.

Runs only with torchvulkan installed and a Vulkan device; each group of tests also needs the
torchvulkan ops it covers. The eager reference runs on the CPU, where every op rounds exactly, so
int8_linear must match bit for bit.
"""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import rope as eager_rope
from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear
from comfy_kitchen.registry import registry

torchvulkan = pytest.importorskip("torchvulkan")
if not torchvulkan.is_available():
    pytest.skip("no Vulkan device", allow_module_level=True)

DEVICE = "vulkan"
needs_int8 = pytest.mark.skipif(not hasattr(torch.ops.torchvulkan, "int8_scaled_mm"),
                                reason="this torchvulkan has no fused int8 ops")
needs_rope = pytest.mark.skipif(not hasattr(torch.ops.torchvulkan, "rms_norm_rope"),
                                reason="this torchvulkan has no rms_norm_rope")


def _dtypes():
    dtypes = [torch.float32, torch.float16]
    if torchvulkan.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


@needs_int8
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


@needs_int8
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


def _freqs(shape, g):
    angles = torch.rand(*shape, generator=g) * 6.28
    c, s = torch.cos(angles), torch.sin(angles)
    return torch.stack([c, -s, s, c], -1).reshape(*shape, 2, 2)


# one rounding step of each dtype: the GPU sums the squares in a different order
ROPE_TOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}


@needs_rope
@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("x_shape,freqs_shape", [
    ((1, 300, 6, 128), (1, 300, 1, 64)),   # [batch, tokens, heads, dim] (Z-Image)
    ((2, 4, 77, 64), (2, 1, 77, 32)),      # [batch, heads, tokens, dim] (Flux)
])
@pytest.mark.parametrize("split_half", [False, True])
def test_rms_rope_matches_eager(dtype, x_shape, freqs_shape, split_half):
    g = torch.Generator().manual_seed(x_shape[1] + x_shape[3])
    q = (torch.randn(x_shape, generator=g) * 2).to(dtype)
    k = (torch.randn(x_shape, generator=g) * 2).to(dtype)
    q_scale = torch.rand(x_shape[-1], generator=g).to(dtype)
    k_scale = torch.rand(x_shape[-1], generator=g).to(dtype)
    freqs = _freqs(freqs_shape, g)
    eager = eager_rope.rms_rope_split_half if split_half else eager_rope.rms_rope
    fused = ck.rms_rope_split_half if split_half else ck.rms_rope
    expected = eager(q, k, freqs, q_scale, k_scale, 1e-6)
    result = fused(q.to(DEVICE), k.to(DEVICE), freqs.to(DEVICE), q_scale.to(DEVICE), k_scale.to(DEVICE), 1e-6)
    for r, e in zip(result, expected, strict=True):
        assert r.dtype == dtype and r.shape == e.shape
        torch.testing.assert_close(r.cpu().float(), e.float(), rtol=ROPE_TOL[dtype], atol=ROPE_TOL[dtype])


@needs_rope
@pytest.mark.parametrize("dtype", _dtypes())
def test_rms_rope1_trims_extra_positions(dtype):
    # the interleaved rope uses the first positions of a longer freqs_cis, as the eager one does
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 2, 40, 64, generator=g).to(dtype)
    scale = torch.rand(64, generator=g).to(dtype)
    freqs = _freqs((1, 1, 50, 32), g)
    expected = eager_rope.rms_rope1(x, freqs, scale, 1e-6)
    result = ck.rms_rope1(x.to(DEVICE), freqs.to(DEVICE), scale.to(DEVICE), 1e-6)
    torch.testing.assert_close(result.cpu().float(), expected.float(), rtol=ROPE_TOL[dtype], atol=ROPE_TOL[dtype])


@needs_rope
def test_rms_rope_backend_selection():
    q = torch.randn(1, 8, 2, 16, device=DEVICE)
    kwargs = {"q": q, "k": q, "freqs_cis": _freqs((1, 8, 1, 8), torch.Generator()).to(DEVICE),
              "q_scale": torch.ones(16, device=DEVICE), "k_scale": None, "epsilon": 1e-6, "rot_dim": 0}
    assert registry.get_capable_backend("rms_rope_split_half", kwargs) == "vulkan"
    # a partial rotation is left to the eager backend
    assert registry.get_capable_backend("rms_rope_split_half", dict(kwargs, rot_dim=8)) != "vulkan"
