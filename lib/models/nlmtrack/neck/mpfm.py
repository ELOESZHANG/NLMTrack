
import torch.nn as nn
import math
import torch
import warnings
from typing import List, Optional
import torch
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        dim=768,
        out_channels=256,
        scale_factors=(0.5, 2.0, 4.0),
        top_block=None,
        norm="LN",
        square_pad=1024,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()
        # assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        self.stages_dec = []
        # input_shapes = net.output_shape()
        # strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        # _assert_strides_are_log2_contiguous(strides)
        strides = [2, 4, 8]
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            # elif scale == 1.0:
            #     layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_dim,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=LayerNorm(out_dim),
                    ),
                    # Conv2d(
                    #     out_channels,
                    #     out_channels,
                    #     kernel_size=3,
                    #     padding=1,
                    #     bias=use_bias,
                    #     norm=LayerNorm(out_channels),
                    # ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)
        # self.upsample = nn.Upsample()
        self.upsample = nn.Upsample(scale_factor=2,  mode='bilinear')
        self.up_stage1_1 = Conv2d(dim*2, dim, kernel_size=1, bias=use_bias, norm=LayerNorm(dim),)
        self.up_stage1_2 = Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias, norm=LayerNorm(dim), )
        self.up_stage1_3 = Conv2d(dim, dim // 2, kernel_size=1, bias=use_bias, norm=LayerNorm(dim // 2), )

        self.up_stage2_1 = Conv2d(dim, dim // 2, kernel_size=1, bias=use_bias, norm=LayerNorm(dim // 2), )
        self.up_stage2_2 = Conv2d(dim//2, dim//2, kernel_size=3, padding=1, bias=use_bias, norm=LayerNorm(dim // 2), )
        self.up_stage2_3 = Conv2d(dim // 2, dim // 4, kernel_size=1, bias=use_bias, norm=LayerNorm(dim // 4), )

        self.down_stage2_1 = Conv2d(dim // 2, dim // 4, kernel_size=1, bias=use_bias, norm=LayerNorm(dim // 4), )
        self.down_stage2_2 = Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, bias=use_bias,
                                  norm=LayerNorm(dim // 4), )
        self.down_stage2_3 = Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, stride=2, bias=use_bias,
                                  norm=LayerNorm(dim // 4), )

        self.down_stage1_2 = Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=use_bias,
                                  norm=LayerNorm(dim // 2), )
        self.down_stage1_3 = Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=2, bias=use_bias,
                                  norm=LayerNorm(dim // 2), )

        self.down = Conv2d(dim*2 + dim // 2, 256,  kernel_size=1, bias=use_bias,
                                  norm=LayerNorm(256), )
        # self.down = Conv2d(dim + dim // 2, 256,  kernel_size=1, bias=use_bias,
        #                           norm=LayerNorm(256), )
        self.down2 = Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias,
                                  norm=LayerNorm(256), )
        # self.down = Conv2d(dim*2 + dim // 2, dim+dim//4,  kernel_size=1, bias=use_bias,
        #                           norm=LayerNorm(dim+dim//4), )
        # self.down2 = Conv2d(dim+dim//4, dim+dim//4, kernel_size=3, padding=1, bias=use_bias,
        #                           norm=LayerNorm(dim+dim//4), )
        # self.down3 = Conv2d(dim+dim//4, (dim+dim//4)//2,  kernel_size=1, bias=use_bias,
        #                           norm=LayerNorm((dim+dim//4)//2), )
        # self.down4 = Conv2d((dim+dim//4)//2, (dim+dim//4)//2, kernel_size=3, padding=1, bias=use_bias,
        #                           norm=LayerNorm((dim+dim//4)//2), )




        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}

        self._square_pad = square_pad



    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        features = x
        results = []

        for stage in self.stages:
            results.append(stage(features))
        # print(results[0].shape,results[1].shape,results[2].shape)

        f_maxp = torch.cat((self.upsample(results[0]), features), dim=1)
        up_stage1 = self.up_stage1_3(self.up_stage1_2(self.up_stage1_1(f_maxp)))
        # up_stage1 = self.up_stage1_3(self.upsample(features))
        up_stage2 = self.up_stage2_3(
            self.up_stage2_2(self.up_stage2_1(torch.cat((self.upsample(up_stage1), results[1]), dim=1))))

        down_stage2 = self.down_stage2_3(self.down_stage2_2(self.down_stage2_1(torch.cat((self.upsample(up_stage2), results[2]), dim=1))))
        down_stage1 = self.down_stage1_3(self.down_stage1_2(torch.cat((down_stage2, up_stage2), dim=1)))
        dec_out = self.down2(self.down(torch.cat((down_stage1, f_maxp), dim=1)))
        return dec_out.flatten(2).transpose(1, 2)   # B,C,H,W -> B,C,N -> B,N,C


def build_neck(cfg):
    model = SimpleFeaturePyramid()
    return model

