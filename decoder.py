# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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
# Adapted from Diffusers' ltx2_diffusion_decoder.py; modified for ComfyUI.
# See THIRD_PARTY_NOTICES.md for the source revision and changes.
"""LTX-2.5 neighborhood attention without an eager quadratic mask allocation."""

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers import LTX2VideoDiffusionDecoderModel
from comfy import model_management
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodAttention,
    LTX2VideoVaeNeighborhoodNattenProcessor,
)

try:
    from natten.functional import na3d
except ImportError:
    na3d = None


compiled_mask = torch.compile(create_block_mask, fullgraph=True)
compiled_attention = torch.compile(flex_attention, fullgraph=True)


def neighborhood_mask(frames, height, width, kernel, device):
    kt, kh, kw = kernel
    hw = height * width

    def mask(b, h, q, k):
        qt, qr = q // hw, q % hw
        qh, qw = qr // width, qr % width
        kt_idx, kr = k // hw, k % hw
        kh_idx, kw_idx = kr // width, kr % width
        st = torch.clamp(qt - kt // 2, 0, frames - kt)
        sh = torch.clamp(qh - kh // 2, 0, height - kh)
        sw = torch.clamp(qw - kw // 2, 0, width - kw)
        return ((kt_idx >= st) & (kt_idx < st + kt) & (kh_idx >= sh)
                & (kh_idx < sh + kh) & (kw_idx >= sw) & (kw_idx < sw + kw))

    tokens = frames * height * width
    return compiled_mask(mask, None, None, tokens, tokens, device=device)


class FlexNeighborhood:
    def __call__(self, attn, hidden_states, block_mask=None):
        shape = hidden_states.shape
        q, k, v = attn.project_qkv(hidden_states)
        q, k, v = (x.flatten(1, 3).transpose(1, 2) for x in (q, k, v))
        out = compiled_attention(q, k, v, block_mask=block_mask, scale=1.0)
        return attn.to_out[0](out.transpose(1, 2).reshape(shape))


class LocalNatten(LTX2VideoVaeNeighborhoodNattenProcessor):
    def __init__(self):
        if na3d is None:
            raise ImportError("NATTEN is not installed. Select the flex decoder backend.")
        self._na3d = na3d
        self.backend = None


class NeighborhoodAttention(LTX2VideoVaeNeighborhoodAttention):
    def __init__(self, dim, kernel_size, head_dim, backend):
        super().__init__(dim, kernel_size, head_dim)
        self.use_flex = backend == "flex"
        self.set_processor(FlexNeighborhood() if self.use_flex else LocalNatten())

    def forward(self, hidden_states, block_mask=None):
        model_management.throw_exception_if_processing_interrupted()
        return super().forward(hidden_states, block_mask)

    def build_block_mask(self, hidden_states):
        model_management.throw_exception_if_processing_interrupted()
        if not self.use_flex:
            return None
        return neighborhood_mask(*hidden_states.shape[1:4], self.kernel_size, hidden_states.device)


def make_decoder(config, backend):
    model = LTX2VideoDiffusionDecoderModel.from_config(config)
    for name, module in list(model.named_modules()):
        if isinstance(module, LTX2VideoVaeNeighborhoodAttention):
            replacement = NeighborhoodAttention(module.heads * module.head_dim,
                                                module.kernel_size, module.head_dim, backend)
            model.set_submodule(name, replacement)
    return model
