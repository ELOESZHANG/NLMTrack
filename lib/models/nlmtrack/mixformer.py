from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from einops import rearrange
from timm.models.layers import DropPath, Mlp
from itertools import repeat
import collections.abc
from lib.utils.misc import is_main_process
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.nlmtrack.pos_utils import get_2d_sincos_pos_embed
from timm.models.registry import register_model

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w*2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_test(self, x, s_h, s_w):
        B, N, C = x.shape
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, _, _ = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        _, k, v = qkv.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_online(self, x, t_h, t_w):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def forward_test(self, x, s_h, s_w):
        x = x + self.drop_path1(self.attn.forward_test(self.norm1(x), s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        x = x + self.drop_path1(self.attn.set_online(self.norm1(x), t_h, t_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size_s=256, img_size_t=128, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__(img_size=224, patch_size=patch_size, in_chans=in_chans,
                                                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, weight_init=weight_init,
                                                norm_layer=norm_layer, act_layer=act_layer)

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])

        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_search = self.grid_size_s ** 2
        self.num_patches_template = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim), requires_grad=False)

        self.init_pos_embed()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_template ** .5),
                                            cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_search ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, images_list):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(images_list[0])  # BCHW-->BNC （1，64，768）
        x_ot = self.patch_embed(images_list[1])   #（1，64，768）
        x_s = self.patch_embed(images_list[2])  # （1，324，768）
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, H_t, W_t, H_s, W_s)

# 层归一化
        x = self.norm(x)  # B,N,C
        return [x]
        # x_t, x_ot, x_s = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s], dim=1)
        #
        # x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        # x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        # x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
        #
        # return x_t_2d, x_ot_2d, x_s_2d

    def forward_test(self, x):
        x = self.patch_embed(x)
        H_s = W_s = self.grid_size_s

        x = x + self.pos_embed_s
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk.forward_test(x, H_s, W_s)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return self.template, x

    def set_online(self, x_t, x_ot):
        x_t = self.patch_embed(x_t)
        x_ot = self.patch_embed(x_ot)

        H_t = W_t = self.grid_size_t

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_ot = x_ot.reshape(1, -1, x_ot.size(-1))  # [1, num_ot * H_t * W_t, C]
        x = torch.cat([x_t, x_ot], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk.set_online(x, H_t, W_t)

        x_t = x[:, :H_t * W_t]
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=H_t, w=W_t)

        self.template = x_t

@register_model
def vit_base_patch16(pretrained=False, pretrain_type='default',
                         search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        img_size_s=search_size, img_size_t=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # if pretrained:
    #     ckpt_path = ''
    #     ckpt = torch.load(ckpt_path, map_location='cpu')['net']
    #     new_dict = {}
    #     for k, v in ckpt.items():
    #         if 'backbone' in k:    # use fixed pos embed
    #             new_dict[k[9:]] = v
    #     missing_keys, unexpected_keys = model.load_state_dict(new_dict, strict=False)
    #     if is_main_process():
    #         print("Load pretrained backbone checkpoint from:", ckpt_path)
    #         print("missing keys:", missing_keys)
    #         print("unexpected keys:", unexpected_keys)

    return model



