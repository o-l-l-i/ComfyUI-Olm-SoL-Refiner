import json
from pathlib import Path

import folder_paths

from .runtime import Refiner


def model_roots():
    roots = folder_paths.get_folder_paths("diffusers")
    roots += folder_paths.get_folder_paths("sol_refiner") if "sol_refiner" in folder_paths.folder_names_and_paths else []
    roots.append(str(Path(folder_paths.models_dir) / "sol_refiner"))
    return list(dict.fromkeys(Path(root).resolve() for root in roots))


def model_names():
    return sorted({child.name for root in model_roots() if root.is_dir()
                   for child in root.iterdir() if child.is_dir() and (child / "model_index.json").is_file()
                   and "sol" in child.name.lower()})


def resolve_model(name):
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Select a model folder from the SoL loader list")
    for root in model_roots():
        directory = (root / name).resolve()
        if not directory.is_relative_to(root):
            continue
        index = directory / "model_index.json"
        if index.is_file():
            with index.open(encoding="utf-8") as f:
                config = json.load(f)
            if config.get("_class_name") != "SoLRefinerH3Pipeline":
                raise ValueError("Select the SoL-Refiner-LTX-2.5-for-MiniMax-H3 package")
            return directory
    raise FileNotFoundError(f"SoL model folder is unavailable: {name}")


class SoLRefinerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (model_names(),),
            "precision": (["fp8", "bf16", "nvfp4"], {"default": "fp8"}),
            "decoder_backend": (["flex", "natten"], {"default": "flex"}),
        }, "optional": {
            "keep_models_in_ram": ("BOOLEAN", {"default": False,
                "tooltip": "Keep CPU model weights for faster reruns. Uses tens of GB of system RAM. GPU weights are offloaded after each stage and run."}),
        }}

    RETURN_TYPES = ("OLM_SOL_REFINER",)
    FUNCTION = "load"
    CATEGORY = "Olm/SoL Refiner"

    def load(self, model_name, precision, decoder_backend, keep_models_in_ram=False):
        if precision not in ("fp8", "bf16", "nvfp4") or decoder_backend not in ("flex", "natten"):
            raise ValueError("Invalid precision or decoder backend")
        return (Refiner(resolve_model(model_name), precision, decoder_backend, keep_models_in_ram),)


class SoLRefineVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "refiner": ("OLM_SOL_REFINER",),
            "images": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
            "width": ("INT", {"default": 1920, "min": 224, "max": 4096, "step": 2}),
            "height": ("INT", {"default": 1080, "min": 224, "max": 4096, "step": 2}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            "decoder_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            "decoder_tile_size": ([256, 384, 512, 768], {"default": 384}),
            "decoder_tile_frames": ([16, 32, 64, 128], {"default": 32}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "refine"
    CATEGORY = "Olm/SoL Refiner"

    def refine(self, refiner, images, prompt, fps, width, height, seed, decoder_seed,
               decoder_tile_size, decoder_tile_frames):
        if decoder_tile_size not in (256, 384, 512, 768) or decoder_tile_frames not in (16, 32, 64, 128):
            raise ValueError("Select a supported decoder tile size")
        return (refiner.refine(images, prompt, fps, width, height, seed, decoder_seed,
                               decoder_tile_size, decoder_tile_frames),)


NODE_CLASS_MAPPINGS = {"OlmSoLRefinerLoader": SoLRefinerLoader, "OlmSoLRefineVideo": SoLRefineVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"OlmSoLRefinerLoader": "SoL Refiner Loader (Olm)",
                              "OlmSoLRefineVideo": "SoL Refine Video (Olm)"}
