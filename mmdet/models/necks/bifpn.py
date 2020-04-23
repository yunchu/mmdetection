import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.core import auto_fp16

from ..registry import NECKS


class PointwiseConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.point_wise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                          padding=0, dilation=1, groups=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.point_wise(x)
        out = self.bn(out)

        return out


class DepthwiseSeparableConv3x3Bn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.depth_wise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                                          padding=1, dilation=1, groups=in_channels)
        self.point_wise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                          padding=0, dilation=1, groups=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        out = self.bn(out)
        return out


class BiFPNBlock(nn.Module):

    def __init__(self, input_channels, num_backbone_features, num_outs, channels):
        super().__init__()
        self.epsilon = 0.0001

        self.channels = channels
        self.num_outs = num_outs
        self.num_backbone_features = num_backbone_features

        assert self.num_outs >= 3

        self.input_to_td_convs = nn.ModuleList()
        self.input_to_output_convs = nn.ModuleList()
        self.top_down_fusion_convs = nn.ModuleList()
        self.bottom_up_fusion_convs = nn.ModuleList()

        self.input_to_td_convs_w = nn.ParameterList()
        self.input_to_output_convs_w = nn.ParameterList()
        self.top_down_upsample_w = nn.ParameterList()
        self.bottom_up_downsample_w = nn.ParameterList()
        self.td_to_output_identities_w = nn.ParameterList()

        for i in range(self.num_outs - 1):
            self.top_down_upsample_w.append(nn.Parameter(torch.ones(1)))
            self.bottom_up_downsample_w.append(nn.Parameter(torch.ones(1)))
            self.td_to_output_identities_w.append(nn.Parameter(torch.ones(1)))

        for i in range(self.num_outs - 1):
            self.input_to_td_convs.append(PointwiseConvBn(input_channels[i], channels))
            self.input_to_td_convs_w.append(nn.Parameter(torch.ones(1)))

        for i in range(self.num_backbone_features - 1):
            self.input_to_output_convs.append(PointwiseConvBn(input_channels[i + 1], channels))
        for _ in range(self.num_outs - 2):
            self.input_to_output_convs_w.append(nn.Parameter(torch.ones(1)))

        for _ in range(self.num_outs - 1):
            self.top_down_fusion_convs.append(DepthwiseSeparableConv3x3Bn(channels, channels))
            self.bottom_up_fusion_convs.append(DepthwiseSeparableConv3x3Bn(channels, channels))

    def forward(self, inputs):

        top_down = []
        top_down_weights_sum = []

        # Compute lateral convolutions on input
        for i in range(self.num_outs - 1):
            w = F.relu(self.input_to_td_convs_w[i])
            top_down.append(w * self.input_to_td_convs[i](inputs[i]))
            top_down_weights_sum.append(w)
        top_down.append(inputs[-1])

        # Compute top-down path
        for i in range(len(top_down) - 1, 0, -1):
            w = F.relu(self.top_down_upsample_w[i - 1])
            top_down[i - 1] = top_down[i - 1] + w * F.interpolate(top_down[i], scale_factor=2, mode='nearest')
            top_down_weights_sum[i - 1] = top_down_weights_sum[i - 1] + w
            top_down[i - 1] = self.top_down_fusion_convs[i - 1](
                top_down[i - 1] / (top_down_weights_sum[i - 1] + self.epsilon))

        bottom_up = [top_down[0], ]
        bottom_up_weights_sum = [None, ]
        # Compute lateral convolutions on top-down
        for i in range(1, self.num_outs):
            w = F.relu(self.td_to_output_identities_w[i - 1])
            bottom_up.append(w * top_down[i])
            bottom_up_weights_sum.append(w)

        # Compute bottom-up path
        for i in range(1, self.num_outs):
            w = F.relu(self.bottom_up_downsample_w[i - 1])
            bottom_up[i] = bottom_up[i] + w * F.max_pool2d(bottom_up[i - 1], kernel_size=3, stride=2, padding=1)
            bottom_up_weights_sum[i] = bottom_up_weights_sum[i] + w

            if i < self.num_outs - 1:
                w = F.relu(self.input_to_output_convs_w[i - 1])
                if i < self.num_backbone_features:
                    bottom_up[i] = bottom_up[i] + w * self.input_to_output_convs[i - 1](inputs[i])
                else:
                    bottom_up[i] = bottom_up[i] + w * inputs[i]
                bottom_up_weights_sum[i] = bottom_up_weights_sum[i] + w

            bottom_up[i] = self.bottom_up_fusion_convs[i - 1](bottom_up[i] / (bottom_up_weights_sum[i] + self.epsilon))

        return tuple(bottom_up)


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 num_outs):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.num_backbone_features = len(in_channels)

        assert self.num_backbone_features >= 2
        assert num_outs - self.num_backbone_features >= 2

        self.in_channels = in_channels
        self.num_outs = num_outs

        self.extra_convs = nn.ModuleList()
        for i in range(self.num_outs - self.num_backbone_features):
            if i == 0:
                input_channels = in_channels[-1]
            else:
                input_channels = out_channels
            if input_channels != out_channels:
                self.extra_convs.append(
                    torch.nn.Sequential(
                        PointwiseConvBn(input_channels, out_channels),
                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    )
                )
            else:
                self.extra_convs.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_channels = in_channels + [out_channels, ] * (self.num_outs - self.num_backbone_features)
            else:
                input_channels = [out_channels, ] * self.num_outs
            self.layers.append(
                BiFPNBlock(input_channels, self.num_backbone_features, self.num_outs, out_channels)
            )

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

        outputs = inputs + extra_inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return tuple(outputs)
