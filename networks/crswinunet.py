from collections import OrderedDict
from ignite.metrics import SSIM
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .swin_transformer_v2 import *
from .denseunet import DenseUNet

class CRSwinUNet(nn.Module):

    def __init__(
        self, 
        img_size=256, 
        patch_size=4,
        in_chans=3, 
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24],
        window_size=7, 
        mlp_ratio=4., 
        qkv_bias=True,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1, 
        ape=False, 
        patch_norm=True,
        attn_skip=False,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False, 
        pretrained_window_sizes=[0, 0, 0, 0], 
        **kwargs
        ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.attn_skip = attn_skip

        self.conv_first = nn.Conv2d(in_chans, embed_dim*2//3, 3, 1, 1)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim*2//3, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.enc_layers = nn.ModuleList()
        self.linear_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.linear_down.append(nn.Linear(2*embed_dim, embed_dim))
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                upsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size // (2 ** i_layer),
                patch_size=patch_size,
                attn_in=False,
                attn_out=attn_skip if (i_layer < self.num_layers - 1) else False
            )
            self.enc_layers.append(layer)

        self.norm = norm_layer(embed_dim)

        self.dec_layers = nn.ModuleList()
        self.linear_up = nn.ModuleList()
        self.linear_cat = nn.ModuleList()
        for i_layer in range(self.num_layers-1, -1, -1):
            self.linear_up.append(nn.Linear(embed_dim // 2, embed_dim))
            self.linear_cat.append(nn.Linear(2*embed_dim, embed_dim))
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                upsample=PatchExpanding if (i_layer > 0) else None,
                use_checkpoint=use_checkpoint,
                img_size=img_size // (2 ** i_layer),
                patch_size=patch_size,
                attn_in=attn_skip if (i_layer < self.num_layers - 1) else False,
                attn_out=False
            )
            self.dec_layers.append(layer)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, out_chans=embed_dim*2//3,
            norm_layer=norm_layer if self.patch_norm else None
        )

        self.norm_out = norm_layer(embed_dim)

        self.conv_after_body = nn.Conv2d(embed_dim*2//3, embed_dim*2//3, 3, 1, 1)

        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.conv_last = nn.Conv2d(embed_dim*2//3, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):

        # print(x.shape)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        enc = list()
        enc.append(x)
        attn_skips = list()
        for i in range(len(self.enc_layers)):
            if enc[-1].shape[-1] > self.embed_dim:
                enc[-1] = self.linear_down[i](enc[-1])
            temp_enc = self.enc_layers[i](enc[-1])
            if self.attn_skip and type(temp_enc) == tuple:
                attn_skips.append(temp_enc[1])
                temp_enc = temp_enc[0]
            enc.append(temp_enc)

        dec = enc[-1]
        dec = self.norm(dec)
        for i in range(len(self.dec_layers)):
            j = len(self.dec_layers) - i - 1
            if i > 0:
                dec = torch.cat((dec, enc[j]), dim=-1)
                if dec.shape[-1] > self.embed_dim:
                    dec = self.linear_cat[i](dec)
            if self.attn_skip and i > 0:
                dec = self.dec_layers[i](dec, prev_attn=attn_skips[j])
            else:
                dec = self.dec_layers[i](dec)
            if dec.shape[-1] < self.embed_dim:
                dec = self.linear_up[i](dec)


        dec = self.norm_out(dec)
        out = self.patch_unembed(dec)

        return out

    def forward(self, x):
        
        _x = x
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        # x = self.conv_after_body(self.forward_features(x))
        # x = self.relu(x)
        out = self.conv_last(x) + _x

        return out

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
