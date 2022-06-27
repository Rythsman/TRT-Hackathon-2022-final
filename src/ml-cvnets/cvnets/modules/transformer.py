#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional

from ..layers import get_normalization_layer, LinearLayer, get_activation_fn, MultiHeadAttention, Dropout
from ..modules import BaseModule
from ..misc.profiler import module_profile
import torch
from os.path import join as pjoin
from collections import OrderedDict
import numpy as np


class TransformerEncoder(BaseModule):
    """
        This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    """
    def __init__(self, opts, embed_dim: int, ffn_latent_dim: int, num_heads: Optional[int] = 8, attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.1, ffn_dropout: Optional[float] = 0.0,
                 transformer_norm_layer: Optional[str] = "layer_norm",
                 *args, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            self.build_act_layer(opts=opts),
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout

    @staticmethod
    def build_act_layer(opts):
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(act_type=act_type, inplace=inplace, negative_slope=neg_slope,
                                      num_parameters=1)
        return act_layer

    def forward(self, x: Tensor) -> Tensor:

        # Multi-head attention
        x = x + self.pre_norm_mha(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        b_sz, seq_len = input.shape[:2]

        out, p_mha, m_mha = module_profile(module=self.pre_norm_mha, x=input)

        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)
        m_ffn = (m_ffn * b_sz * seq_len)

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs

    def load_from_np_weight(self, weights, n_block):
        
        def np2th(tensor):
            return torch.from_numpy(tensor)

        ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
        ATTENTION_K = "MultiHeadDotProductAttention_1/key"
        ATTENTION_V = "MultiHeadDotProductAttention_1/value"
        ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
        FC_0 = "MlpBlock_3/Dense_0"
        FC_1 = "MlpBlock_3/Dense_1"
        ATTENTION_NORM = "LayerNorm_0"
        MLP_NORM = "LayerNorm_2"

        with torch.no_grad():

            ROOT = f"Transformer/encoderblock_{n_block}"

            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.embed_dim, self.embed_dim).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.embed_dim, self.embed_dim).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.embed_dim, self.embed_dim).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.embed_dim, self.embed_dim).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
            qkv_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)
            self.pre_norm_mha[1].qkv_proj.weight.copy_(qkv_weight)
            self.pre_norm_mha[1].qkv_proj.bias.copy_(qkv_bias)
            self.pre_norm_mha[1].out_proj.weight.copy_(out_weight)
            self.pre_norm_mha[1].out_proj.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            mlp1 = self.pre_norm_ffn[1]
            mlp2 = self.pre_norm_ffn[-2]
            mlp1.weight.copy_(mlp_weight_0)
            mlp2.weight.copy_(mlp_weight_1)
            mlp1.bias.copy_(mlp_bias_0)
            mlp2.bias.copy_(mlp_bias_1)

            norm1 = self.pre_norm_mha[0]
            norm2 = self.pre_norm_ffn[0]
            norm1.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            norm1.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            norm2.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            norm2.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


    def export_npz_dict(self):
        ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
        ATTENTION_K = "MultiHeadDotProductAttention_1/key"
        ATTENTION_V = "MultiHeadDotProductAttention_1/value"
        ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
        FC_0 = "MlpBlock_3/Dense_0"
        FC_1 = "MlpBlock_3/Dense_1"
        ATTENTION_NORM = "LayerNorm_0"
        MLP_NORM = "LayerNorm_2"
        
        # ROOT = f"Transformer/encoderblock_{n_block}"
        ret_dict = OrderedDict()
        
        with torch.no_grad():
            # [out_channel, in_channel]
            qkv_weight = self.pre_norm_mha[1].qkv_proj.weight.detach().cpu().numpy()
            qkv_bias = self.pre_norm_mha[1].qkv_proj.bias.detach().cpu().numpy().reshape(-1)

            out_weight = self.pre_norm_mha[1].out_proj.weight.detach().cpu().numpy()
            out_bias = self.pre_norm_mha[1].out_proj.bias.detach().cpu().numpy().reshape(-1)
            q_weight, k_weight, v_weight = np.split(qkv_weight, 3, axis=0)
            
            query_weight = np.transpose(q_weight, [1, 0])
            key_weight = np.transpose(k_weight, [1, 0])
            value_weight = np.transpose(v_weight, [1, 0])
            out_weight = np.transpose(out_weight, [1, 0])
            print('has tranposed')

            query_bias, key_bias, value_bias = np.split(qkv_bias, 3, axis=0)

            mlp1 = self.pre_norm_ffn[1]
            mlp2 = self.pre_norm_ffn[-2]
            mlp_weight_0 = mlp1.weight.detach().t().cpu().numpy()
            mlp_weight_1 = mlp2.weight.detach().t().cpu().numpy()
            mlp_bias_0 = mlp1.bias.detach().cpu().numpy().reshape(-1)
            mlp_bias_1 = mlp2.bias.detach().cpu().numpy().reshape(-1)
            
            norm1 = self.pre_norm_mha[0]
            norm2 = self.pre_norm_ffn[0]
            attention_norm_weight = norm1.weight.view(-1).detach().cpu().numpy()
            attention_norm_bias = norm1.bias.view(-1).detach().cpu().numpy()
            ffn_norm_weight = norm2.weight.view(-1).detach().cpu().numpy()
            ffn_norm_bias = norm2.bias.view(-1).detach().cpu().numpy()

            ret_dict[pjoin(ATTENTION_Q, "kernel")] = query_weight
            ret_dict[pjoin(ATTENTION_K, "kernel")] = key_weight
            ret_dict[pjoin(ATTENTION_V, "kernel")] = value_weight
            ret_dict[pjoin(ATTENTION_OUT, "kernel")] = out_weight

            ret_dict[pjoin(ATTENTION_Q, "bias")] = query_bias
            ret_dict[pjoin(ATTENTION_K, "bias")] = key_bias
            ret_dict[pjoin(ATTENTION_V, "bias")] = value_bias
            ret_dict[pjoin(ATTENTION_OUT, "bias")] = out_bias

            ret_dict[pjoin(FC_0, "kernel")] = mlp_weight_0
            ret_dict[pjoin(FC_1, "kernel")] = mlp_weight_1
            ret_dict[pjoin(FC_0, "bias")] = mlp_bias_0
            ret_dict[pjoin(FC_1, "bias")] = mlp_bias_1

            ret_dict[pjoin(ATTENTION_NORM, "scale")] = attention_norm_weight
            ret_dict[pjoin(ATTENTION_NORM, "bias")] = attention_norm_bias
            ret_dict[pjoin(MLP_NORM, "scale")] = ffn_norm_weight
            ret_dict[pjoin(MLP_NORM, "bias")] = ffn_norm_bias

            keys = list(ret_dict.keys())
            keys.sort()

            new_ret_dict = OrderedDict()
            for k in keys:
                new_ret_dict[k] = np.ascontiguousarray(ret_dict[k])

            return new_ret_dict

        """
            ROOT = f"Transformer/encoderblock_{n_block}"
            with torch.no_grad():
                query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

                query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
                key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
                value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
                out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

                self.attn.query.weight.copy_(query_weight)
                self.attn.key.weight.copy_(key_weight)
                self.attn.value.weight.copy_(value_weight)
                self.attn.out.weight.copy_(out_weight)
                self.attn.query.bias.copy_(query_bias)
                self.attn.key.bias.copy_(key_bias)
                self.attn.value.bias.copy_(value_bias)
                self.attn.out.bias.copy_(out_bias)

                mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
                mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
                mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
                mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

                self.ffn.fc1.weight.copy_(mlp_weight_0)
                self.ffn.fc2.weight.copy_(mlp_weight_1)
                self.ffn.fc1.bias.copy_(mlp_bias_0)
                self.ffn.fc2.bias.copy_(mlp_bias_1)

                self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
                self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
                self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
                self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
        """
