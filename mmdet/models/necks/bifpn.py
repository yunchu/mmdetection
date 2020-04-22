import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.core import auto_fp16

from ..registry import NECKS
from ..utils import ConvModule


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1,
                                   groups=input_channels)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class BiFPNBlock(nn.Module):

    def __init__(self, num_outs, channels, conv_cfg, norm_cfg, activation):
        super().__init__()
        self.channels = channels
        self.num_outs = num_outs
        self.activation = activation

        assert self.num_outs >= 3

        self.lateral_convs_from_input = nn.ModuleList()
        self.lateral_convs_from_top_down = nn.ModuleList()

        for _ in range(self.num_outs - 1):
            conv = ConvModule(
                channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lateral_convs_from_input.append(conv)

        for _ in range(self.num_outs - 1):
            conv = ConvModule(
                channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lateral_convs_from_top_down.append(conv)

    def forward(self, inputs):
        outputs = [
            self.lateral_convs_from_input[i](input) if i < self.num_outs - 1 else input
            for i, input in enumerate(inputs)
        ]

        for i in range(len(outputs) - 1, 0, -1):
            outputs[i - 1] += F.interpolate(outputs[i], scale_factor=2, mode='nearest')

        outputs = [
            self.lateral_convs_from_top_down[i - 1](output) if i > 0 else output
            for i, output in enumerate(outputs)
        ]

        for i in range(1, len(outputs)):
            outputs[i] += F.interpolate(outputs[i - 1], scale_factor=0.5, mode='nearest')

        for i in range(1, len(outputs) - 1):
            outputs[i] += inputs[i]

        return tuple(outputs)


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 num_outs,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        assert num_outs >= 3
        assert num_outs >= len(in_channels)

        self.in_channels = in_channels
        self.num_outs = num_outs
        self.activation = activation

        self.channel_equlizers = nn.ModuleList()
        for input_channels in in_channels:
            conv = DepthwiseSeparableConv(input_channels, out_channels)
            # TODO(ikrylov): bn, actication?
            self.channel_equlizers.append(conv)

        self.extra_convs = nn.ModuleList()
        for i in range(self.num_outs - len(self.in_channels)):
            if i == 0:
                input_channels = in_channels[-1]
            else:
                input_channels = out_channels
            extra_fpn_conv = ConvModule(
                input_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.extra_convs.append(extra_fpn_conv)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                BiFPNBlock(num_outs, out_channels, conv_cfg, norm_cfg, self.activation)
            )
            in_channels = [out_channels, ] * len(self.in_channels)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
        extra_inputs = []
        for i in range(self.num_outs - len(self.in_channels)):
            if i == 0:
                extra_inputs.append(self.extra_convs[i](inputs[-1]))
            else:
                extra_inputs.append(self.extra_convs[i](extra_inputs[-1]))

        outputs = [self.channel_equlizers[i](input) for i, input in enumerate(inputs)]
        outputs.extend(extra_inputs)

        for layer in self.layers:
            outputs = layer(outputs)

        return tuple(outputs)
