import torch
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
        self.epsilon = 0.0001

        self.channels = channels
        self.num_outs = num_outs
        self.activation = activation

        assert self.num_outs >= 3

        self.lc_from_input = nn.ModuleList()
        self.lc_from_top_down = nn.ModuleList()
        self.fc_top_down = nn.ModuleList()
        self.fc_bottom_up = nn.ModuleList()

        self.lc_from_input_w = nn.ParameterList()
        self.lc_from_top_down_w = nn.ParameterList()
        self.top_down_w = nn.ParameterList()
        self.bottom_up_w = nn.ParameterList()
        self.shortcuts_w = nn.ParameterList()
        for i in range(self.num_outs - 1):
            self.top_down_w.append(nn.Parameter(torch.ones(1)))
            self.bottom_up_w.append(nn.Parameter(torch.ones(1)))
            if i > 0:
                self.shortcuts_w.append(nn.Parameter(torch.ones(1)))

        for _ in range(self.num_outs - 1):
            conv = ConvModule(
                channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lc_from_input.append(conv)
            self.lc_from_input_w.append(nn.Parameter(torch.ones(1)))

        for _ in range(self.num_outs - 1):
            conv = ConvModule(
                channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lc_from_top_down.append(conv)
            self.lc_from_top_down_w.append(nn.Parameter(torch.ones(1)))

        for _ in range(self.num_outs - 1):
            self.fc_top_down.append(DepthwiseSeparableConv(channels, channels))
            self.fc_bottom_up.append(DepthwiseSeparableConv(channels, channels))

    def forward(self, inputs):

        top_down = []
        top_down_weights_sum = []

        # Compute lateral convolutions on input
        for i in range(self.num_outs - 1):
            w = F.relu(self.lc_from_input_w[i])
            top_down.append(w * self.lc_from_input[i](inputs[i]))
            top_down_weights_sum.append(w)
        top_down.append(inputs[-1])

        # Compute top-down path
        for i in range(len(top_down) - 1, 0, -1):
            w = F.relu(self.top_down_w[i - 1])
            top_down[i - 1] = top_down[i - 1] + w * F.interpolate(top_down[i], scale_factor=2,
                                                                  mode='nearest')
            top_down_weights_sum[i - 1] = top_down_weights_sum[i - 1] + w

        # Fuse lateral and top-down
        for i in range(self.num_outs - 1):
            top_down[i] = self.fc_top_down[i](
                top_down[i] / (top_down_weights_sum[i] + self.epsilon))

        bottom_up = [top_down[0], ]
        bottom_up_weights_sum = [None, ]
        # Compute lateral convolutions on top-down
        for i in range(1, self.num_outs):
            w = F.relu(self.lc_from_top_down_w[i - 1])
            bottom_up.append(w * self.lc_from_top_down[i - 1](top_down[i]))
            bottom_up_weights_sum.append(w)

        # Compute bottom-up path
        for i in range(1, len(bottom_up)):
            w = F.relu(self.bottom_up_w[i - 1])
            bottom_up[i] = bottom_up[i] + w * F.interpolate(bottom_up[i - 1], scale_factor=0.5,
                                                            mode='nearest')
            bottom_up_weights_sum[i] = bottom_up_weights_sum[i] + w

        # Add shortcuts
        for i in range(1, len(bottom_up) - 1):
            w = F.relu(self.shortcuts_w[i - 1])
            bottom_up[i] = bottom_up[i] + w * inputs[i]
            bottom_up_weights_sum[i] = bottom_up_weights_sum[i] + w

        # Fuse
        for i in range(1, len(bottom_up)):
            bottom_up[i] = self.fc_bottom_up[i - 1](
                bottom_up[i] / (bottom_up_weights_sum[i] + self.epsilon))

        return tuple(bottom_up)


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
