import torch.nn as nn
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class ExtraStages(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 add_output_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 debug=False):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        assert isinstance(out_channels, list)
        self.out_channels = out_channels
        self.activation = activation
        self.add_output_convs = add_output_convs
        self.debug = debug
        self.fp16_enabled = False

        assert len(self.out_channels) == 1 or len(self.out_channels) == num_outs
        if len(self.out_channels) == 1:
            self.out_channels = self.out_channels * num_outs

        self.extra_stages = nn.ModuleList()

        if num_outs >= 1:
            in_channels = self.in_channels[-1]
            for i in range(num_outs):
                extra_conv = ConvModule(
                    in_channels,
                    out_channels[i],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.extra_stages.append(extra_conv)
                in_channels = out_channels[i]

        self.extra_convs = nn.ModuleList()
        if self.add_output_convs:
            for c in self.in_channels + self.out_channels:
                self.extra_convs.append(
                    ConvModule(
                        c,
                        c,
                        1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False)
                )

        # self.extra_stages = nn.Sequential(*self.extra_stages)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):

        for x, c in zip(inputs, self.in_channels):
            assert x.shape[1] == c

        x = inputs[-1]

        extras = []
        for extra_stage in self.extra_stages:
            if self.debug:
                print(extra_stage)
            x = extra_stage(x)
            extras.append(x)

        outputs = inputs + extras

        if self.add_output_convs:
            for i, (x, conv) in enumerate(zip(outputs, self.extra_convs)):
                outputs[i] = conv(x)

        outputs = tuple(outputs)

        if self.debug:
            for i, o in enumerate(outputs):
                print(i, o.shape)

        return outputs
