import logging
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKLLTX2Video, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.video_processor import VideoProcessor
from transformers import AutoModelForImageTextToText, AutoTokenizer

from comfy import model_management, utils

from .decoder import make_decoder
from .loading import component_patcher, load_component, prepare_text_operations, read_config
from .models import VideoTransformer, use_comfy_attention


SIGMA = 0.9093750119
logger = logging.getLogger(__name__)


def output_geometry(width, height, frames):
    if width < 224 or height < 224:
        raise ValueError("LTX-2.5 decoding needs a canvas of at least 224 x 224 pixels")
    if frames < 1:
        raise ValueError("Provide at least one video frame")
    return math.ceil(width / 64) * 64, math.ceil(height / 64) * 64, max(17, 1 + math.ceil((frames - 1) / 8) * 8)


class Refiner:
    def __init__(self, directory, precision, decoder_backend, keep_models_in_ram=False):
        self.directory = Path(directory)
        self.precision = precision
        self.decoder_backend = decoder_backend
        self.keep_models_in_ram = keep_models_in_ram
        self.components = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory / "tokenizer", local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def component(self, name):
        if name in self.components:
            return self.components[name]
        model_management.throw_exception_if_processing_interrupted()
        logger.info("SoL Refiner: loading %s", name)
        directory = self.directory / name
        config = read_config(directory)
        if name == "text_encoder":
            model = AutoModelForImageTextToText.from_pretrained(
                directory, local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="olm_sol_comfy", low_cpu_mem_usage=True,
            ).model
            prepare_text_operations(model)
            model.eval()
            patcher = component_patcher(model)
        else:
            with torch.device("meta"):
                if name == "transformer":
                    model = VideoTransformer.from_config(config)
                elif name == "vae":
                    model = AutoencoderKLLTX2Video.from_config(config)
                    model.decoder = None
                elif name == "connectors":
                    model = LTX2TextConnectors.from_config(config)
                elif name == "latent_upsampler":
                    model = LTX2LatentUpsamplerModel.from_config(config)
                elif name == "diffusion_decoder":
                    model = make_decoder(config, self.decoder_backend)
                else:
                    raise ValueError(f"Unknown component: {name}")
            patcher = load_component(model, directory, self.precision if name == "transformer" else "bf16")
            use_comfy_attention(model)
        self.components[name] = patcher
        return patcher

    def activate(self, name, working_memory):
        self.offload()
        if name != "transformer" and model_management.vram_state == model_management.VRAMState.NO_VRAM:
            raise ValueError("SoL's text and video components require full component loads. Restart ComfyUI without --novram.")
        patcher = self.component(name)
        # Only the video transformer implements partial execution offload for
        # all its weights. The upstream components also own ordinary buffers
        # and parameters that require a complete load before execution.
        model_management.load_models_gpu([patcher], memory_required=working_memory,
                                         force_full_load=name != "transformer")
        return patcher.model.inner

    def offload(self):
        for patcher in self.components.values():
            if patcher.loaded_size() > 0:
                model_management.unload_model_and_clones(patcher)

    def refine(self, images, prompt, fps, width, height, seed, decoder_seed, tile_size, tile_frames):
        try:
            return self._refine(images, prompt, fps, width, height, seed, decoder_seed, tile_size, tile_frames)
        finally:
            self.offload()
            if not self.keep_models_in_ram:
                self.components.clear()

    def _refine(self, images, prompt, fps, width, height, seed, decoder_seed, tile_size, tile_frames):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Provide a batch of RGB video frames")
        if not prompt.strip():
            raise ValueError("Provide a prompt describing the video")
        cw, ch, frames = output_geometry(width, height, images.shape[0])
        device = model_management.get_torch_device()
        progress = utils.ProgressBar(54)

        logger.info("SoL Refiner: encoding prompt")
        text_model = self.activate("text_encoder", 3 * 1024**3)
        tokens = self.tokenizer([prompt.strip()], padding="max_length", max_length=1024,
                                truncation=True, add_special_tokens=True, return_tensors="pt").to(device)
        encoded = text_model(**tokens, output_hidden_states=True, use_cache=False)
        embeddings = torch.stack(encoded.hidden_states, dim=-1).flatten(2, 3).to("cpu")
        mask = tokens.attention_mask.to("cpu")
        del encoded, tokens
        progress.update_absolute(1)

        connectors = self.activate("connectors", 3 * 1024**3)
        context, _, mask = connectors(embeddings.to(device), mask.to(device), padding_side="left")
        context, mask = context.to("cpu"), mask.to("cpu")
        del embeddings
        progress.update_absolute(2)

        source_frames = images.shape[0]
        if frames != source_frames:
            images = torch.cat((images, images[-1:].expand(frames - source_frames, -1, -1, -1)))
        video = images.permute(0, 3, 1, 2).unsqueeze(0)
        pixels = self.video_processor.preprocess_video(video, height=ch // 2, width=cw // 2)
        logger.info("SoL Refiner: encoding %d frames", source_frames)
        vae = self.activate("vae", 8 * 1024**3)
        vae.enable_tiling()
        latent = vae.encode(pixels.to(device=device, dtype=torch.bfloat16)).latent_dist.mode().to("cpu")
        mean, std = vae.latents_mean.to("cpu"), vae.latents_std.to("cpu")
        scale = vae.config.scaling_factor
        del pixels, video
        progress.update_absolute(3)

        upsampler = self.activate("latent_upsampler", 4 * 1024**3)
        latent = upsampler(latent.to(device))
        latent = LTX2ConditionPipeline._normalize_latents(latent, mean, std, scale).to("cpu")
        progress.update_absolute(4)

        logger.info("SoL Refiner: one-step refinement (%s)", self.precision)
        transformer = self.activate("transformer", 8 * 1024**3)
        latent = latent.to(device)
        noise = torch.randn(latent.shape, device=device, dtype=latent.dtype,
                            generator=torch.Generator(device).manual_seed(seed))
        noisy = ((1 - SIGMA) * latent.float() + SIGMA * noise.float()).to(torch.bfloat16)
        del noise, latent
        velocity = transformer(noisy, context.to(device), mask.to(device), SIGMA, fps,
                               lambda done, total: progress.update_absolute(4 + done))
        latent = (noisy.float() - SIGMA * velocity.float()).to("cpu")
        del noisy, velocity, context, mask
        progress.update_absolute(52)

        logger.info("SoL Refiner: decoding with %s attention", self.decoder_backend)
        decoder = self.activate("diffusion_decoder", 16 * 1024**3)
        decoder.enable_tiling(tile_sample_min_height=tile_size, tile_sample_min_width=tile_size,
                              tile_sample_stride_height=tile_size * 2 // 3 // 8 * 8,
                              tile_sample_stride_width=tile_size * 2 // 3 // 8 * 8,
                              tile_sample_min_num_frames=tile_frames,
                              tile_sample_stride_num_frames=tile_frames * 2 // 3 // 2 * 2)
        latent = LTX2ConditionPipeline._denormalize_latents(latent, mean, std, scale)
        decoded = decoder.decode(latent.to(device=device, dtype=torch.bfloat16),
                                 generator=torch.Generator(device).manual_seed(decoder_seed)).sample
        top, left = (ch - height) // 2, (cw - width) // 2
        result = decoded[0, :, :source_frames, top:top + height, left:left + width]
        result = ((result.float() + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0).contiguous().cpu()
        progress.update_absolute(54)
        return result
