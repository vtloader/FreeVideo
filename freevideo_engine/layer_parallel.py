"""Hold part of the block stack on each accelerator, and hand the activation over.

The layers are serial -- block 25 needs block 24's output -- so splitting them buys no
speed. What it buys is room: each rank materializes only its own contiguous range of
transformer blocks, so a model that does not fit one 61 GiB device (this one is 65.7 GB)
fits two, instead of being streamed through host memory every step.

The activation crosses once at the boundary and the result crosses back once. At the
measured 61 GiB/s that is about 0.15% of a step, and it is the only traffic.

The forward is a copy of the model's own, patched on the way install_ulysses patches
its own, so no file in the model source changes. The last rank runs the output head and
sends the result back, which leaves both ranks holding the same real outputs: the
sampling loop then stays in lockstep, and the engine's existing rule that rank 0 owns
the decode and every shared file is untouched.
"""
from __future__ import annotations

import os
import types

import torch

# FREEVIDEO_LAYER_DEBUG=1 prints the placement of every tensor a block receives. The
# first split run died inside a block's normalisation with a host tensor where it
# expected the device, and the placement of each input is what names the culprit.
_debug = os.environ.get('FREEVIDEO_LAYER_DEBUG', '') not in ('', '0')


def ranges(total, world):
    """The contiguous block ranges, the remainder rounded onto the earlier ranks."""
    out = []
    start = 0
    for index in range(world):
        count = total // world + (1 if index < total % world else 0)
        out.append((start, start + count))
        start += count
    return out


def range_for(rank, world, total):
    return ranges(total, world)[rank]


def _dist():
    import torch.distributed as dist
    return dist


def _send(tensor, dst):
    _dist().send(tensor.contiguous(), dst=dst)


def _recv_like(src, like):
    """Receive a tensor whose shape the peer announces first.

    The receiver has not run the blocks that produce it, so it cannot infer the shape
    from anything local; one small message carries it. ``like`` is a tensor or a device:
    the buffers must already be where the collectives run, and on the rank that receives
    the activation there is no local tensor of that shape to copy the placement from.
    """
    where = getattr(like, 'device', like)
    header = torch.empty(3, dtype=torch.int64, device=where)
    _dist().recv(header, src=src)
    rows, width, dtype_code = (int(value) for value in header.tolist())
    dtype = torch.bfloat16 if dtype_code == 0 else torch.float16
    buffer = torch.empty((1, rows, width), dtype=dtype, device=where)
    _dist().recv(buffer, src=src)
    return buffer


def _send_like(tensor, dst):
    header = torch.tensor([tensor.shape[1], tensor.shape[2],
                           0 if tensor.dtype == torch.bfloat16 else 1],
                          dtype=torch.int64, device=tensor.device)
    _send(header, dst)
    _send(tensor, dst)


def _forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
             timestep_indices, token_tags, position_ids, video_indices, audio_indices,
             text_indices, attention_kwargs=None, return_dict=True):
    """The model's forward, running only this rank's blocks."""
    # Imported here, not at module scope: diffusers is the vendored tree, which only
    # becomes importable once paths.add_vdn() has run. This module is imported by the
    # engine's own startup, before that.
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerOutput)
    if torch.is_grad_enabled():
        raise RuntimeError('layer-parallel forward is inference-only; use torch.no_grad()')
    if attention_kwargs and attention_kwargs.get('scale', 1.0) != 1.0:
        raise ValueError('merge LoRA weights before splitting layers; runtime LoRA scale is unsupported')
    rank, first, last = self._layer_range

    rotary_emb = self.rope(position_ids)
    embed_device = next(self.time_embedder.parameters()).device
    temb = self.time_embedder(self.time_proj(timestep).to(
        device=embed_device, dtype=get_parameter_dtype(self.time_embedder)))
    adaln_indices = (timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags).to(embed_device)

    if first == 0:
        # The projections live on the device; the inputs may arrive on the host (the
        # prompt embeddings do), so move as well as recast.
        host = self.proj_in.weight.device
        video_embeds = self.proj_in(
            hidden_states.to(device=host, dtype=get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(
            audio_hidden_states.to(device=host, dtype=get_parameter_dtype(self.audio_proj_in)))
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(device=host, dtype=get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)
        packed = text_embeds.new_zeros((text_embeds.shape[0], position_ids.shape[0],
                                        text_embeds.shape[-1]))
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
    else:
        packed = _recv_like(rank - 1, embed_device)

    for index in range(first, last):
        block = self.transformer_blocks[index]
        if _debug:
            weight = next(block.parameters(), None)
            print('  rank %d block %d packed=%s/%s temb=%s adaln=%s rotary=%s weight=%s'
                  % (rank, index, packed.device, packed.dtype, temb.device, adaln_indices.device,
                     rotary_emb[0].device, weight.device if weight is not None else 'none'),
                  flush=True)
        packed = block(packed, temb, adaln_indices, rotary_emb)

    if last < len(self.transformer_blocks):
        # Hand the activation to the next stage and wait for the finished result.
        _send_like(packed, rank + 1)
        video_output = _recv_like(rank + 1, packed)
        audio_output = _recv_like(rank + 1, packed)
    else:
        packed = self.norm_out(packed, temb, timestep_indices.to(embed_device)).to(
            get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(packed).index_select(1, video_indices)
        audio_output = self.audio_proj_out(packed).index_select(1, audio_indices)
        _send_like(video_output, rank - 1)
        _send_like(audio_output, rank - 1)

    if not return_dict:
        return video_output, audio_output
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def install(model, rank, world):
    """Patch the model to run only rank's blocks. Returns that range."""
    total = len(model.transformer_blocks)
    first, last = range_for(rank, world, total)
    model._layer_range = (rank, first, last)
    model.forward = types.MethodType(_forward, model)
    return first, last
