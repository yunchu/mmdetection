# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import torch
import torchvision
import torch.nn as nn
from mmcv.cnn import (constant_init, kaiming_init, normal_init)
from ..builder import BACKBONES


@BACKBONES.register_module()
class PTMobilenetV2Wrapper(nn.Module):
    def __init__(self, output_features):
        super(PTMobilenetV2Wrapper, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=False)
        self.output_features = output_features
        self.model.features = self.model.features[0 : max(output_features) + 1]
        self.model.classifier = None

    def init_weights(self, pretrained=None):
        chpt = torch.load(pretrained)['model']
        chpt = {k[len('module.model.model.'):] if 'module.model.model.' in k else '' : v for k,v in chpt.items()}
        self.model.load_state_dict(chpt, strict=False)

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.model.features):
            x = block(x)
            if i in self.output_features:
                outs.append(x)

        return tuple(outs)
