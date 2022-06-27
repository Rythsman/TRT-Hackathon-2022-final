#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, OrderedDict, Tuple

from .transformer import TransformerEncoder
from .base_module import BaseModule
from ..misc.profiler import module_profile
from ..layers import ConvLayer, get_normalization_layer
from os.path import join as pjoin


class MobileViTBlock(BaseModule):
    """
        MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, opts, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32, attn_dropout: Optional[float] = 0.1,
                 dropout: Optional[int] = 0.1, ffn_dropout: Optional[int] = 0.1, patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8, transformer_norm_layer: Optional[str] = "layer_norm",
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, var_ffn: Optional[bool] = False,
                 no_fusion: Optional[bool] = False,
                 *args, **kwargs):
        conv_3x3_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation
        )
        conv_1x1_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )

        conv_1x1_out = ConvLayer(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                opts=opts, in_channels=2 * in_channels, out_channels=in_channels,
                kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True
            )
        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(opts=opts, embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout,
                               transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.var_ffn = var_ffn
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tconv_in_dim={}, conv_out_dim={}, dilation={}, conv_ksize={}".format(self.cnn_in_dim, self.cnn_out_dim, self.dilation, self.conv_ksize)
        repr_str += "\n\tpatch_h={}, patch_w={}".format(self.patch_h, self.patch_w)
        repr_str += "\n\ttransformer_in_dim={}, transformer_n_heads={}, transformer_ffn_dim={}, dropout={}, " \
                    "ffn_dropout={}, attn_dropout={}, blocks={}".format(
            self.cnn_out_dim,
            self.n_heads,
            self.ffn_dim,
            self.dropout,
            self.ffn_dropout,
            self.attn_dropout,
            self.n_blocks
        )
        if self.var_ffn:
            repr_str += "\n\t var_ffn_min_mult={}, var_ffn_max_mult={}".format(
                self.ffn_min_dim, self.ffn_max_dim
            )

        repr_str += "\n)"
        return repr_str

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = map(int, feature_map.shape)

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(-1, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(-1, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(-1, num_patches, in_channels)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        last_dim = (patches.numel() // info_dict["batch_size"] // self.patch_area // info_dict["total_patches"])
        patches = patches.contiguous().view(-1, self.patch_area, info_dict["total_patches"], last_dim)

        batch_size, pixels, num_patches, channels = map(int, patches.size())
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(-1, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(-1, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(
                torch.cat((res, fm), dim=1)
            )
        return fm

    def export_npz(self, filepath):
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
        
        all_npz_dict = OrderedDict()
        for i, transformer_encoder in enumerate(self.global_rep[0:-1]):
            ROOT = f"Transformer/encoderblock_{i}"
            npz_dict = transformer_encoder.export_npz_dict()
            for k in npz_dict:
                new_k = pjoin(ROOT, k)
                all_npz_dict[new_k] = npz_dict[k]

        post_norm = self.global_rep[-1]

        all_npz_dict['Transformer/encoder_norm/scale'] = post_norm.weight.detach().cpu().view(-1).numpy()
        all_npz_dict['Transformer/encoder_norm/bias'] = post_norm.bias.detach().cpu().view(-1).numpy()

        np.savez(filepath, **all_npz_dict)
        

    def load_from_npz(self, npz_path):
        
        np_weight = np.load(npz_path)

        for i, transformer_encoder in enumerate(self.global_rep[0:-1]):
            transformer_encoder.load_from_np_weight(np_weight, i)
        
        post_norm = self.global_rep[-1] 
        post_norm.weight.copy_(torch.from_numpy(np_weight['Transformer/encoder_norm/scale']))
        post_norm.bias.copy_(torch.from_numpy(np_weight['Transformer/encoder_norm/bias']))


    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = macs = 0.0

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, info_dict = self.unfolding(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding(patches=patches, info_dict=info_dict)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        if self.fusion is not None:
            out, p, m = module_profile(module=self.fusion, x=torch.cat((out, res), dim=1))
            params += p
            macs += m

        return res, params, macs