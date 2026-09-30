# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Diffusers' transformer_ltx2.py; modified for ComfyUI.
# See THIRD_PARTY_NOTICES.md for the source revision and changes.
"""Video-only execution of the released Diffusers LTX-2.5 components."""

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS

from comfy import model_management, ops
from comfy.ldm.modules.attention import optimized_attention
from diffusers import LTX2VideoTransformer3DModel
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2VideoTransformerBlock,
    apply_split_rotary_emb,
)


def comfy_text_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    model_management.throw_exception_if_processing_interrupted()
    out = optimized_attention(query, key, value, query.shape[1], mask=attention_mask,
                              skip_reshape=True, skip_output_reshape=True,
                              scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("olm_sol_comfy", comfy_text_attention)
AttentionMaskInterface.register("olm_sol_comfy", ALL_MASK_ATTENTION_FUNCTIONS["eager"])


class ComfyLTXAttention:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, query_rotary_emb=None, key_rotary_emb=None):
        model_management.throw_exception_if_processing_interrupted()
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = attn.norm_q(attn.to_q(hidden_states))
        k = attn.norm_k(attn.to_k(context))
        v = attn.to_v(context)
        if query_rotary_emb is not None:
            q = apply_split_rotary_emb(q, query_rotary_emb)
            k = apply_split_rotary_emb(k, query_rotary_emb if key_rotary_emb is None else key_rotary_emb)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(q.dtype))[:, None, None, :] * -10000.0
        out = optimized_attention(q, k, v, attn.heads, mask=attention_mask)
        if attn.to_gate_logits is not None:
            gates = 2 * torch.sigmoid(attn.to_gate_logits(hidden_states))
            out = (out.unflatten(-1, (attn.heads, -1)) * gates.unsqueeze(-1)).flatten(-2)
        return attn.to_out[0](out)


class VideoBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        for name in ("norm1", "norm2", "norm3", "attn1", "attn2", "ff",
                     "scale_shift_table", "prompt_scale_shift_table"):
            setattr(self, name, getattr(block, name))

    def forward(self, x, context, temb, prompt_temb, rope, mask):
        table = ops.cast_to_input(self.scale_shift_table, x)
        shift, scale, gate, ff_shift, ff_scale, ff_gate, q_shift, q_scale, q_gate = (
            LTX2VideoTransformerBlock.get_mod_params(table, temb, x.shape[0])
        )
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.attn1(h, query_rotary_emb=rope) * gate
        p_shift, p_scale = LTX2VideoTransformerBlock.get_mod_params(
            ops.cast_to_input(self.prompt_scale_shift_table, x), prompt_temb, x.shape[0]
        )
        h = self.norm2(x) * (1 + q_scale) + q_shift
        x = x + self.attn2(h, context * (1 + p_scale) + p_shift, attention_mask=mask) * q_gate
        return x + self.ff(self.norm3(x) * (1 + ff_scale) + ff_shift) * ff_gate


class VideoTransformer(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ("proj_in", "proj_out", "norm_out", "rope", "time_embed",
                     "prompt_adaln", "scale_shift_table"):
            setattr(self, name, getattr(original, name))
        self.transformer_blocks = nn.ModuleList(VideoBlock(b) for b in original.transformer_blocks)

    @classmethod
    def from_config(cls, config):
        if not (config["cross_attn_mod"] and config["use_prompt_adaln_single"]
                and config["rope_type"] == "split" and config["patch_size"] == 1
                and config["patch_size_t"] == 1):
            raise ValueError("This loader requires the released SoL LTX-2.5/H3 transformer configuration")
        return cls(LTX2VideoTransformer3DModel.from_config(config))

    def forward(self, latent, context, mask, sigma, fps, progress=None):
        batch, channels, frames, height, width = latent.shape
        x = latent.flatten(2).transpose(1, 2)
        coords = self.rope.prepare_video_coords(batch, frames, height, width, x.device, fps=fps)
        rope = self.rope(coords, device=x.device)
        x = self.proj_in(x)
        timestep = torch.full((batch,), sigma * 1000, device=x.device, dtype=torch.float32)
        temb, embedded = self.time_embed(timestep, batch_size=batch, hidden_dtype=x.dtype)
        prompt_temb, _ = self.prompt_adaln(timestep, batch_size=batch, hidden_dtype=x.dtype)
        temb, prompt_temb = temb[:, None], prompt_temb[:, None]
        for i, block in enumerate(self.transformer_blocks):
            model_management.throw_exception_if_processing_interrupted()
            x = block(x, context, temb, prompt_temb, rope, mask)
            if progress is not None:
                progress(i + 1, len(self.transformer_blocks))
        shift, scale = (ops.cast_to_input(self.scale_shift_table, x)[None, None] + embedded[:, None, None]).unbind(2)
        x = self.proj_out(self.norm_out(x) * (1 + scale) + shift)
        return x.transpose(1, 2).reshape(batch, channels, frames, height, width)


def use_comfy_attention(model):
    for module in model.modules():
        if isinstance(module, LTX2Attention):
            module.set_processor(ComfyLTXAttention())
