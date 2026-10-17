"""Short, isolated consumer-GPU compatibility checks for doctor --probe."""
from .torch_device import device_module, device_name
import json
import math
import sys
import platform
import traceback
import torch
from .locking import runtime_lock

# Ascend runs the same small shapes through Torch-NPU's own operators in BF16, so
# the limits below are that path's measured spread rather than the CUDA ones.
# Measured on CANN 9.1 / Ascend910 with these shapes: BF16 GEMM against an FP32
# reference 1.7e-3, attention through Torch-NPU's scaled dot product 3.2e-5.
NPU_LINEAR_TOLERANCE = .01
NPU_ATTENTION_TOLERANCE = .001
# Ascend reads a quantized cache by decoding it to BF16 and running BF16, so these
# bound what a cache read may cost before the run rather than what a quantized
# product costs. Measured against the unquantized weight: int8 ConvRot rows 0.9%,
# E4M3 rows 2.6%; the limits leave room for the BF16 matrix on top.
NPU_INT8_STORAGE_TOLERANCE = .03
NPU_FP8_STORAGE_TOLERANCE = .05


def flash_attention_available():
    """The CUDA Flash Attention question, or None where this build has none to ask."""
    try:
        return torch.backends.cuda.is_flash_attention_available()
    except Exception:
        return None


@torch.no_grad()
def main():
    backend = sys.argv[1]
    device = device_name(torch)
    torch.set_num_threads(4)
    torch.manual_seed(171)
    with runtime_lock():
        if backend == 'linear' and device == 'npu':
            # Ascend plans BF16 weights, so the bounded GEMM shape the CUDA probe
            # uses runs here against an FP32 reference: one finite result with a
            # bounded error, and no model weights loaded.
            x = torch.randn(701, 512, device=device, dtype=torch.bfloat16)
            weight = torch.randn(12288, 512, device=device, dtype=torch.bfloat16)
            shapes = {'activation': list(x.shape), 'weight': list(weight.shape), 'output_dtype': str(x.dtype)}
            value = x @ weight.t()
            reference = x.float() @ weight.float().t()
            policy = 'bf16'
            tolerance = NPU_LINEAR_TOLERANCE
        elif backend == 'linear':
            from .paths import add_vdn
            add_vdn()
            from src.models.ops.fp8_linear import quantize_activation, per_tensor_gemm
            from .fp8_gemm import scaled_mm
            from .kernel_capabilities import fp8_implementation
            # Wide output and a tail tile exercise the bounded compatibility
            # epilogue on the user's real GPU without loading any model.
            x = torch.randn(701, 512, device='cuda', dtype=torch.bfloat16)
            weight = torch.randn(12288, 512, device='cuda', dtype=torch.bfloat16)
            shapes = {'activation': list(x.shape), 'weight': list(weight.shape), 'output_dtype': str(x.dtype)}
            if torch.cuda.get_device_capability() < (8, 9):
                from .weight_only import linear
                scale = weight.float().abs().amax().reshape(1, 1) / 448.
                fp8 = (weight.float() / scale).to(torch.float8_e4m3fn)
                value = linear(x, fp8, scale)
                policy = 'bf16-weight-only'
            else:
                scale = (weight.float().abs().amax() if per_tensor_gemm() else weight.float().abs().amax(1)).reshape(1, -1) / 448.
                fp8 = (weight.float() / scale.reshape(-1, 1)).to(torch.float8_e4m3fn)
                xq, xs = quantize_activation(x)
                implementation = fp8_implementation(platform.system(), 'per_tensor' if per_tensor_gemm() else 'rowwise')
                value = scaled_mm(xq, fp8.t(), xs, scale, out_dtype=x.dtype, implementation=implementation)
                policy = 'native-fp8-per-tensor' if per_tensor_gemm() else 'native-fp8-rowwise'
                policy += '/' + implementation
            # Small dequantized reference only; no model weights are loaded.
            reference_x = x.float() if torch.cuda.get_device_capability() < (8, 9) else xq.float() * xs
            reference = reference_x @ (fp8.float() * scale.reshape(-1, 1)).t()
            tolerance = .01
        elif backend == 'linear-int8':
            if device == 'npu':
                # Ascend has no int8 product, so this probes what the backend does
                # with a quantized cache instead: backends/npu_linears decodes the
                # storage to BF16 and the linear runs in BF16. Every storage a
                # prepared cache can carry is decoded here, because being unable to
                # read one belongs in the setup checks rather than in the middle of
                # a sampling run.
                from .backends import npu_linears
                group = npu_linears.CONVROT_GROUP
                rows, columns = 1024, 2 * group
                weight = torch.randn(rows, columns, dtype=torch.float32)
                # Written the way an encoder writes them: int8 rows rotate into the
                # ConvRot basis, E4M3 rows take their amax over 448.
                rotated = npu_linears.convrot_rotate(weight, group)
                int8_scale = rotated.abs().amax(1, keepdim=True) / 127.
                per_tensor = weight.abs().amax() / 448.
                per_row = weight.abs().amax(1, keepdim=True) / 448.
                cases = (
                    ('int8-convrot', npu_linears.decode_int8(
                        (rotated / int8_scale).round().clamp(-127, 127).to(torch.int8),
                        int8_scale[:, 0], group), NPU_INT8_STORAGE_TOLERANCE),
                    ('fp8-per-tensor', npu_linears.decode_fp8(
                        (weight / per_tensor).to(torch.float8_e4m3fn), per_tensor.reshape(1)),
                     NPU_FP8_STORAGE_TOLERANCE),
                    ('fp8-rowwise', npu_linears.decode_fp8(
                        (weight / per_row).to(torch.float8_e4m3fn), per_row[:, 0]),
                     NPU_FP8_STORAGE_TOLERANCE))
                x = torch.randn(701, columns, device=device, dtype=torch.bfloat16)
                exact = x.float() @ weight.to(device).t()
                storage = {}
                value = reference = None
                for label, decoded, limit in cases:
                    # The decoder produces host BF16, exactly as the load hook does.
                    actual = x @ decoded.to(device).t()
                    rmse = ((actual.float() - exact).square().mean()
                            / exact.square().mean().clamp_min(1e-20)).sqrt().item()
                    if not math.isfinite(rmse) or rmse > limit:
                        raise RuntimeError(f'Ascend {label} cache read relative RMSE {rmse:.6f} exceeded {limit}')
                    storage[label] = dict(relative_rmse=rmse, limit=limit, decoded_dtype=str(decoded.dtype))
                    if value is None:
                        value, reference = actual, exact
                shapes = {'activation': list(x.shape), 'weight': list(weight.shape), 'storage': storage}
                policy = 'bf16-from-quantized-cache'
                tolerance = cases[0][2]
            else:
                from .int8_ops import GROUP, hadamard16, matmul, rotate_quantize
                # The prepared int8 path on the user's GPU: ConvRot rotation and
                # per-row quantization, then the int8 GEMM against the same rows
                # dequantized in FP32. A tail row block is included; no model loads.
                x = torch.randn(701, 2 * GROUP, device='cuda', dtype=torch.bfloat16)
                weight = torch.randint(-127, 128, (1024, 2 * GROUP), device='cuda', dtype=torch.int8)
                weight_scale = torch.rand(1024, device='cuda') / 64 + 1e-3
                values, scales = rotate_quantize(x)
                h16 = hadamard16(x.device)
                rotated = (h16 @ x.float().reshape(len(x), -1, 16, 16) @ h16).reshape(len(x), -1)
                dequantized = values.float() * scales[:, None]
                rotation = ((dequantized - rotated).square().mean() / rotated.square().mean()).sqrt().item()
                if not math.isfinite(rotation) or rotation > .03:
                    raise RuntimeError(f'Int8 rotation/quantization relative RMSE {rotation:.6f} exceeded 0.03')
                value = matmul((values, scales), weight, weight_scale, out_dtype=torch.float32)
                reference = dequantized @ (weight.float() * weight_scale[:, None]).t()
                shapes = {'activation': list(x.shape), 'weight': list(weight.shape), 'rotation_relative_rmse': rotation}
                policy = 'int8-convrot'
                tolerance = 1e-4
        else:
            if backend == 'torch-flash' and device == 'cuda' and not flash_attention_available():
                raise RuntimeError('This PyTorch wheel was built without CUDA Flash Attention. '
                                   'Automatic selection can use cuDNN or Sage2 if their probes pass.')
            from .attention import WindowAttention
            from .backends import get_backend
            from src.models.sequence_layout import SequenceLayout
            from src.models.softmax_attention.window import window_bounds, window_softmax_reference
            layout = SequenceLayout(seq_len=487, video_start=3, num_frames=15, tokens_per_frame=32)
            q, k, v = [torch.randn(487, 4, 128, device=device, dtype=torch.bfloat16) for _ in range(3)]
            shapes = {'qkv': list(q.shape), 'dtype': str(q.dtype), 'stride': list(q.stride())}
            # CUDA keeps taking the process default, as it always has; another
            # device names its own backend, exactly as the engine's worker does.
            attention = WindowAttention(backend, window_batch=2,
                                        device_backend=get_backend(device) if device != 'cuda' else None)
            value = attention(q, k, v, layout, window_bounds(15, radius=2, chunk=3), 128 ** -.5, 'both')
            policy = dict(attention.backend_calls)
            reference = window_softmax_reference(q, k, v, layout, window_bounds(15, radius=2, chunk=3), 128 ** -.5, 'both')
            tolerance = .06 if backend == 'sage2' else (NPU_ATTENTION_TOLERANCE if device == 'npu' else .006)
        device_module(torch).synchronize()
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError('Kernel returned non-finite values')
        relative = ((value.float() - reference.float()).square().mean() /
                    reference.float().square().mean().clamp_min(1e-20)).sqrt().item()
        if not math.isfinite(relative) or relative > tolerance:
            raise RuntimeError(f'Kernel relative RMSE {relative:.6f} exceeded {tolerance}')
        print(json.dumps({'backend': backend, 'status': 'complete', 'policy': policy,
                          'device': device, 'gpu': device_module(torch).get_device_name(),
                          'capability': list(torch.cuda.get_device_capability()) if device == 'cuda' else None,
                          'scope': 'Small-shape execution only', 'finite': True,
                          'relative_rmse': relative, 'relative_rmse_limit': tolerance,
                          'shapes': shapes,
                          'torch_flash_compiled': flash_attention_available() if device == 'cuda' else None,
                          'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                          'system': platform.system()}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        print(json.dumps({'backend': sys.argv[1], 'status': 'error', 'error': str(error),
                          'torch_flash_compiled': flash_attention_available(),
                          'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                          'system': platform.system()}), flush=True)
        sys.exit(1)
