"""Read local component weights and use ComfyUI's quantized operations."""

import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from comfy import model_management, model_patcher, ops, quant_ops, utils


def read_config(directory):
    with (Path(directory) / "config.json").open(encoding="utf-8") as f:
        return json.load(f)


def component_patcher(model):
    # Transformers/Diffusers expose a read-only device property; ComfyUI owns
    # device metadata on this container and still sees every child parameter.
    container = nn.Module()
    container.add_module("inner", model)
    return model_patcher.ModelPatcher(container, model_management.get_torch_device(), torch.device("cpu"))


def prepare_text_operations(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            replacement = ops.manual_cast.Linear(module.in_features, module.out_features,
                                                 module.bias is not None, device="meta")
        else:
            continue
        replacement.weight = module.weight
        if isinstance(module, nn.Linear):
            replacement.bias = module.bias
        model.set_submodule(name, replacement)


def prepare_operations(model, precision):
    quantized = set()
    mixed = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            quantize = precision != "bf16" and min(module.in_features, module.out_features) >= 128
            # Keep time modulation and boundary projections in BF16.
            quantize = quantize and name.startswith("transformer_blocks.")
            op = mixed.Linear if quantize else ops.manual_cast.Linear
            replacement = op(module.in_features, module.out_features, module.bias is not None,
                             device="cpu" if quantize else "meta", dtype=torch.bfloat16)
            model.set_submodule(name, replacement)
            if quantize:
                quantized.add(name)
        elif isinstance(module, nn.Dropout):
            model.set_submodule(name, nn.Identity())
        elif isinstance(module, nn.RMSNorm):
            model.set_submodule(name, ops.manual_cast.RMSNorm(
                module.normalized_shape, eps=module.eps, elementwise_affine=module.weight is not None,
                device="meta", dtype=torch.bfloat16))
        elif isinstance(module, nn.LayerNorm):
            model.set_submodule(name, ops.manual_cast.LayerNorm(
                module.normalized_shape, eps=module.eps, elementwise_affine=module.weight is not None,
                bias=module.bias is not None, device="meta", dtype=torch.bfloat16))
    return quantized


def load_component(model, directory, precision="bf16"):
    directory = Path(directory)
    expected = dict(model.state_dict())
    quantized = prepare_operations(model, precision)
    state = {}
    device = model_management.get_torch_device()
    if precision != "bf16" and not model_management.supports_fp8_compute(device):
        raise ValueError("FP8 refinement requires an NVIDIA GPU with FP8 support; select bf16 on this device")
    if precision == "nvfp4" and not model_management.supports_nvfp4_compute(device):
        raise ValueError("NVFP4 requires a supported Blackwell GPU")
    layout = quant_ops.TensorCoreNVFP4Layout if precision == "nvfp4" else quant_ops.TensorCoreFP8Layout
    if quantized:
        model_management.free_memory(2 * 1024**3, device)
    quant_format = "nvfp4" if precision == "nvfp4" else "float8_e4m3fn"
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    progress = utils.ProgressBar(len(expected))
    loaded = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in expected:
                    continue
                model_management.throw_exception_if_processing_interrupted()
                value = f.get_tensor(key)
                if tuple(value.shape) != tuple(expected[key].shape):
                    raise ValueError(f"Unexpected shape for {directory.name}/{key}: {value.shape}")
                name = key.removesuffix(".weight")
                if key.endswith(".weight") and name in quantized:
                    q, params = layout.quantize(value.to(device=device, dtype=torch.bfloat16), scale="recalculate")
                    state[key] = q.cpu()
                    state[name + ".weight_scale"] = (params.block_scale if precision == "nvfp4" else params.scale).cpu()
                    if precision == "nvfp4":
                        state[name + ".weight_scale_2"] = params.scale.cpu()
                    state[name + ".comfy_quant"] = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)
                    del q, params
                else:
                    state[key] = value.to(dtype=torch.bfloat16, copy=True)
                loaded += 1
                progress.update_absolute(loaded)
    missing = expected.keys() - state.keys()
    if missing:
        raise ValueError(f"Missing weights in {directory}: {sorted(missing)[:8]}")
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    unresolved = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if unresolved:
        raise ValueError(f"Uninitialized component tensors: {unresolved[:8]}")
    return component_patcher(model)
