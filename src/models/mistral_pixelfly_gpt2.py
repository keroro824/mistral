"""
mistral_gpt2.py

Custom Implementation of the GPT-2 LM-Head Model (and auxiliary classes) with support for adaptive/custom number of
gradient checkpoints (for fine-grained tweaking of memory footprint vs. speed).

Reference: https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP
from src.models.modules.blocksparse_linear import ButterflyBlockSparseLinear

# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.models.gpt2_gc")


class MistralPixelflyGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(
        self,
        config: GPT2Config,
        reorder_attn: bool = True,
        upcast_attn: bool = True,
        gradient_checkpointing: bool = True,
        gc_checkpoint_every: int = 1,
    ):
        super().__init__(config)
        self.reorder_attn, self.upcast_attn = reorder_attn, upcast_attn

        # Turn on Gradient Checkpointing if Necessary
        if gradient_checkpointing:
            self.create_checkpointed_model(gc_checkpoint_every)
        else:
            self.create_model()

    # @MERCURY =>> Reconfigure GPT2LMHead to take custom, partial checkpoint model instance!
    def create_checkpointed_model(self, gc_checkpoint_every: int):
        # Reinitalize GPT-2 Model w/ Custom GC Wrapper
        self.transformer = MistralGPT2Model(self.config, gc_checkpoint_every, self.reorder_attn, self.upcast_attn)

    # @MERCURY =>> Reconfigure GPT2LMHead to Initialize Standard (non-checkpointed) model instance!
    def create_model(self):
        # Reinitialize Custom GPT-2 Model
        self.transformer = MistralGPT2Model(
            self.config, gc_checkpoint_every=-1, reorder_attn=self.reorder_attn, upcast_attn=self.upcast_attn
        )


class MistralGPT2Model(GPT2Model):
    # @MERCURY =>> GPT-2 Model Instance now takes `gc_checkpoint_every` parameter.
    def __init__(self, config: GPT2Config, gc_checkpoint_every: int, reorder_attn: bool, upcast_attn: bool):
        super().__init__(config)
        self.h = nn.ModuleList(
            [
                MistralPixelflyGPT2Block(
                    config.n_ctx, config, i + 1, scale=True, reorder_attn=reorder_attn, upcast_attn=upcast_attn
                )
                for i in range(config.n_layer)
            ]
        )
        self.init_weights()


class MistralPixelflyGPT2MLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        embed_dim = config.hidden_size
        self.c_fc = ButterflyBlockSparseLinear(embed_dim, intermediate_size, blocks=4)
        self.c_proj = ButterflyBlockSparseLinear(intermediate_size, embed_dim, blocks=4)


class MistralPixelflyGPT2Block(GPT2Block):
    def __init__(self, n_ctx, config, layer_num, scale=False, reorder_attn=True, upcast_attn=True):
        super().__init__(config, layer_idx=layer_num)
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.mlp = MistralPixelflyGPT2MLP(inner_dim, config)
        print(self.mlp.c_fc)
