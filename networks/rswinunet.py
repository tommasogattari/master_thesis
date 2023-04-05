from collections import OrderedDict
from ignite.metrics import SSIM
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_v2 import *
from .denseunet import DenseUNet

logger = logging.getLogger(__name__)

class RSwinUnet(nn.Module):
    def __init__(self, 
        config, 
        img_size=224,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 3, 2],
        num_heads=[3, 6, 12, 24],
        attn_skip=False,
        use_checkpoint=False,
        attn_drop_rate=0.,
        qkv_bias=True,
        mlp_ratio=4.,
        window_size=7,
        ape=False,
        drop_path_rate=0.1,
        drop_rate=0.,
        patch_size=4,
        norm_layer=nn.LayerNorm,
        num_classes=21843,
        patch_norm=True,
        zero_head=False,
        vis=False):
        super(RSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.patch_norm=patch_norm
        self.ape=ape
        self.mlp_ratio = mlp_ratio
        self.num_layers = len(depths)
        self.in_chans = in_chans
        self.rswin_unet = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)


        self.conv_first = nn.Conv2d(1, embed_dim*2//3, 3, 1, 1)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position 
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        #dpr = [x.item() for x in torch.linspace(0, float(drop_path_rate), sum(depths))]

        # build layers
        self.enc_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=int(embed_dim * 2 ** i_layer),
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

        self.norm = norm_layer(int(embed_dim * 2 ** i_layer))

        self.dec_layers = nn.ModuleList()
        self.linear_cat = nn.ModuleList()
        for i_layer in range(self.num_layers-1, -1, -1):
            self.linear_cat.append(nn.Linear(
                2*int(embed_dim * 2 ** i_layer),
                int(embed_dim * 2 ** i_layer)
            )  if i_layer != self.num_layers-1 else nn.Identity())
            layer = RSTB(
                dim=int(embed_dim * 2 ** i_layer),
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

        self.conv_after_body = nn.Conv2d(1, embed_dim*2//3, 3, 1, 1)


        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.conv_last = nn.Conv2d(1, embed_dim*2//3, 3, 1, 1)


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
        
        # enc = list()
        # enc.append(x)
        # attn_skips = list()
        # for i in range(len(self.enc_layers)):
        #     if enc[-1].shape[-1] > self.embed_dim:
        #         enc[-1] = self.linear_down[i](enc[-1])
        #     temp_enc = self.enc_layers[i](enc[-1])
        #     if self.attn_skip and type(temp_enc) == tuple:
        #         attn_skips.append(temp_enc[1])
        #         temp_enc = temp_enc[0]
        #     enc.append(temp_enc)

        # dec = enc[-1]
        # dec = self.norm(dec)
        # for i in range(len(self.dec_layers)):
        #     j = len(self.dec_layers) - i - 1
        #     if i > 0:
        #         dec = torch.cat((dec, enc[j]), dim=-1)
        #         if dec.shape[-1] > self.embed_dim:
        #             dec = self.linear_cat[i](dec)
        #     if self.attn_skip and i > 0:
        #         dec = self.dec_layers[i](dec, prev_attn=attn_skips[j])
        #     else:
        #         dec = self.dec_layers[i](dec)
        #     if dec.shape[-1] < self.embed_dim:
        #         dec = self.linear_up[i](dec)

        enc = list()
        enc.append(x)
        attn_skips = list()
        for i in range(len(self.enc_layers)):
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

        dec = self.norm_out(dec)
        out = self.patch_unembed(dec)

        return out

    def forward(self, x):
        
        _x = x
        print("STEP1", x.shape)
        x = self.conv_first(x)
        #print("---------Output size after conv_first:------------", x.shape)
        print("STEP2")
        x = self.conv_after_body(self.forward_features(x)) + x
        print("STEP3")
        #print("-----------Output size after conv_after_body:-----------", x.shape)
        x = self.relu(x)
        print("STEP4")
        #print("-----------Output size after ReLU:-------------", x.shape)
        out = self.conv_last(x) #+ _x
        print("conv_LAAASSSSST")
        #print("-------------Final output size:----------", out.shape)
        return out

    

    # def forward(self, x):
    #     if x.size()[1] == 1:
    #         x = x.repeat(1,3,1,1)
    #     logits = self.rswin_unet(x)
    #     return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.rswin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.rswin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.rswin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops