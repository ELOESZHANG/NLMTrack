"""
NLMTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.utils.misc import NestedTensor
from lib.models.nlmtrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.models.nlmtrack.neck.mpfm import build_neck
from timm.models.layers import trunc_normal_


def visiul(target, size):
    target = (target - min(target)) / (max(target) - min(target))
    target = target * 255
    target = target.reshape(size).unsqueeze(0)
    # resize=size*16
    transform = transforms.Resize(size=(size[0]*16,size[1]*16))
    target = transform(target)
    target = torch.squeeze(transform(target).permute(1, 2, 0))
    # target = target.repeat(3, 1, 1).permute(1, 2, 0)
    target = target.cpu()
    target = target.numpy()
    return np.uint8(target)


class NLMTRACK(nn.Module):
    """ This is the base class for NLMTrack """
    def __init__(self, encoder, decoder, neck, hidden_dim,
                 bins=1000, feature_type='x', num_frames=1, num_template=1):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim
        #self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim) # the bottleneck layer, which aligns the dimmension of encoder and decoder
        self.bottleneck = nn.Linear(480, hidden_dim)
        self.decoder = decoder
        self.vocab_embed = MLP(hidden_dim, hidden_dim, bins+2, 3)

        # neck
        self.neck = neck
        # self.neck_encoder = neck.encoder
        # self.neck_decoder = neck.decoder
        # self.neck_norm = neck.norm

        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

        # Different type of visual features for decoder.
        # Since we only use one search image for now, the 'x' is same with 'x_last' here.
        if self.feature_type == 'x':
            num_patches = self.num_patch_x * self.num_frames
        elif self.feature_type == 'xz':
            num_patches = self.num_patch_x * self.num_frames + self.num_patch_z * self.num_template
        elif self.feature_type == 'token':
            num_patches = 1
        else:
            raise ValueError('illegal feature type')

        # position embeding for the decocder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化neck参数
        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.neck.apply(_init_weights)



    def forward(self, images_list=None, zx=None, seq=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "neck":
            return self.forward_neck(zx)
        elif mode == "decoder":
            return self.forward_decoder(zx, seq)
        else:
            raise ValueError

    def forward_neck(self, zx):
        ''' z_feat: 模板区域； pre_z_feat: 先前模板区域 ； x_feat: 搜索区域特征 '''
        zx = zx[-1]
        zx = zx[:,-self.num_patch_x * self.num_frames:]
        B,N,C = zx.shape
        zx = zx.transpose(1, 2).reshape(B, C, int(N**0.5) , int(N**0.5))
        decoder_feat = self.neck(zx)

        # target = torch.squeeze(z_feat).mean(-1,keepdim=True)
        # target_online = torch.squeeze(pre_z_feat).mean(-1, keepdim=True)
        # search = torch.squeeze(x_feat).mean(-1, keepdim=True)
        # fig,axs = plt.subplots(1,3)
        # axs[0].imshow(visiul(target,(7,7)))
        # axs[1].imshow(visiul(target_online, (14, 14)))
        # axs[2].imshow(visiul(search, (14, 14)))
        # plt.imshow()
        # search = torch.squeeze(decoder_feat).mean(-1, keepdim=True)
        # plt.imshow(visiul(search, (14, 14)))
        return decoder_feat


    def forward_encoder(self, images_list):
        # Forward the encoder
        zx = self.encoder(images_list)
        # target = torch.squeeze(xz[-1][:,:64,:]).mean(-1,keepdim=True)
        # target_online = torch.squeeze(xz[-1][:, 64:128, :]).mean(-1,keepdim=True)
        # search= torch.squeeze(xz[-1][:,128:,:]).mean(-1,keepdim=True)
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(visiul(target,(8,8)))
        # axs[1].imshow(visiul(target_online,(8,8)))
        # axs[2].imshow(visiul(search,(18,18)))
        # plt.show()
        # # imshow
        # target = torch.squeeze(xz[-1][:,:64,:]).mean(-1,keepdim=True)
        # cv2.imshow('target', visiul(target,(8,8)))
        # cv2.waitKey(10000)
        # target_online = torch.squeeze(xz[-1][:, 64:128, :]).mean(-1,keepdim=True)
        # cv2.imshow('target_online', visiul(target_online,(8,8)))
        # cv2.waitKey(10000)
        # search= torch.squeeze(xz[-1][:,128:,:]).mean(-1,keepdim=True)
        # cv2.imshow('search', visiul(search,(18,18)))
        # cv2.waitKey(10000)

        return zx

    def forward_decoder(self, neck_x_feature, sequence):

        # zx_mem = zx[-1]
        dec_mem= neck_x_feature
        B, _, _ = dec_mem.shape
        # # get different type of visual features for decoder.
        # if self.feature_type == 'x': # get features of all search images
        #     dec_mem = zx_mem[:,-self.num_patch_x * self.num_frames:]
        # elif self.feature_type == 'xz': # get all features of search and template images
        #     dec_mem = zx_mem
        # elif self.feature_type == 'token': # get an average feature vector of search and template images.
        #     dec_mem = zx_mem.mean(1).unsqueeze(1)
        # else:
        #     raise ValueError('illegal feature type')

        # align the dimensions of the encoder and decoder
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  #[B,NL,D]
        dec_mem = dec_mem.permute(1,0,2)  #[NL,B,D]

        out = self.decoder(dec_mem, self.pos_embed.permute(1,0,2).expand(-1,B,-1), sequence)
        out = self.vocab_embed(out) # embeddings --> likelihood of words

        return out

    def inference_decoder(self, xz, sequence, window=None, seq_format='xywh'):
        # Forward the decoder
        # xz_mem = xz[-1]
        # B, _, _ = xz_mem.shape

        # # get different type of visual features for decoder.
        # if self.feature_type == 'x':
        #     dec_mem = xz_mem[:,-self.num_patch_x:]
        # elif self.feature_type == 'xz':
        #     dec_mem = xz_mem
        # elif self.feature_type == 'token':
        #     dec_mem = xz_mem.mean(1).unsqueeze(1)
        # else:
        #     raise ValueError('illegal feature type')
        dec_mem= xz
        B, _, _ = dec_mem.shape
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  #[B,NL,D]
        dec_mem = dec_mem.permute(1,0,2)  #[NL,B,D]

        out = self.decoder.inference(dec_mem,
                                    self.pos_embed.permute(1,0,2).expand(-1,B,-1),
                                    sequence, self.vocab_embed,
                                    window, seq_format)

        return out



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_nlmtrack(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    neck = build_neck(cfg)
    model = NLMTRACK(
        encoder,
        decoder,
        neck,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins=cfg.MODEL.BINS,
        feature_type=cfg.MODEL.FEATURE_TYPE,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER
    )

    return model
