# Third-party notices

`models.py` adapts the video path of Diffusers' `transformer_ltx2.py`.
`decoder.py` adapts the neighborhood mask and attention processor from Diffusers'
`ltx2_diffusion_decoder.py`. The transformer is copyright 2025 The Lightricks
team and The HuggingFace Team. The decoder is copyright 2026 Lightricks and
The HuggingFace Team. All rights reserved. Both adapted files, including the
integration changes, are licensed under Apache License 2.0; a copy is in
[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). The remaining original
integration code is covered by [LICENSE.txt](LICENSE.txt). Neither code license
grants rights to model weights or other third-party material.

Changes select ComfyUI attention and model operations, omit isolated audio
transformer branches, compile the decoder mask, and use locally installed NATTEN
when selected. The remaining component implementations are imported from
Diffusers and Transformers.

The adapted Diffusers source is from revision
`e0abab83b5df05de9e7abd788643c1a7c1e42e28`:

- [Video transformer](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/models/transformers/transformer_ltx2.py)
- [Diffusion decoder](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/models/autoencoders/ltx2_diffusion_decoder.py)

That Diffusers revision contains no `NOTICE` file.

The one-step schedule and inference sequence in `runtime.py`, and the local
NATTEN adapter pattern in `decoder.py`, follow NVIDIA's SoL Refiner H3 reference:
https://github.com/NVlabs/Sana/tree/670482d8a857d578ac8a2ea89b052d0fb47badba/models/sol-refiner/MiniMax-H3

That snapshot contains no license file at the repository root or under
`models/sol-refiner`. Apache coverage for this reference snapshot has not been
verified; the license on Sana's separate `main` branch is not treated as proof.

Weights are loaded from the user's local download and are not distributed by
this project. Code licensing does not grant rights to model weights or inputs.

As checked on 2026-09-30, the
[H3 weight repository](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3/tree/4655337474c1c77fb52e9ac55111552e625e779c)
contained no model card or license file, and its Hub metadata declared no
license. Its applicable weight terms could not be established from that
repository. The Apache license for Diffusers code does not resolve this;
consult the model publisher for the applicable model and component terms.

The checkpoint is described as based on LTX-2.5, so the base model's
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
may also be relevant; this does not establish the complete terms for the SoL package.
