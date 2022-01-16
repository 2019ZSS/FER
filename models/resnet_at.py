import copy
import torch
import torch.nn as nn

from .utils import load_state_dict_from_url
from .resnet import BasicBlock, Bottleneck, ResNet, resnet18

from .at import (
    ContextBlock,
    StripPooling,
    BilinearCNN,
    PSAModule,
)

from .attention_gate import(
    AttentionGate,
    AttentionGate2
)

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


def getLayerSetting(resenet_type):
    if resenet_type == 'resnet18':
        return BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]
    if resenet_type == 'resnet34':
        return BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512]
    if resenet_type == 'resnet50':
        return Bottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048]
    raise NotImplementedError('{} not implemented'.format(resenet_type))


def getATModule(at_type):
    if at_type == 'CB':
        return ContextBlock
    if at_type == 'SP':
        return StripPooling 
    if at_type == 'AG':
        return AttentionGate
    if at_type == 'AG2':
        return AttentionGate2
    if at_type == 'PSA':
        return PSAModule
    raise NotImplementedError('{} not implemented'.format(at_type))


def getLowHeadModule(low_head_type):
    if low_head_type == 'PSA':
        return PSAModule
    raise NotImplementedError('{} not implemented'.format(low_head_type))


class ResNetAT(ResNet):

    def __init__(self, at_type, at_kws, low_head_type=None, low_head_kw=None, bc_kw=None, at_layer=[0, 0, 0, 0, 1], resnet_type='resnet18', pretrained=False, in_channels=3, num_classes=1000, drop=0.0):
        if pretrained:
            assert in_channels == 3
        assert resnet_type in ('resnet18', 'resnet34', 'resnet50')
        block, layers, features = getLayerSetting(resnet_type)
        super(ResNetAT, self).__init__(
            block=block, layers=layers, in_channels=in_channels, num_classes=1000 if pretrained else num_classes
        )
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[resnet_type], progress=True)
            self.load_state_dict(state_dict)
        if low_head_type is not None:
            self.low_head_type = low_head_type
            self.low_head_module = getLowHeadModule(low_head_type)(**low_head_kw)
        else:
            self.low_head_type = None
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self._bc_kw = bc_kw
        if bc_kw:
            self.fc = BilinearCNN(**bc_kw)
        else:
            self.fc = nn.Linear(in_features=features[-1], out_features=num_classes)
        self.at_layer = at_layer
        at_model = getATModule(at_type=at_type)
        if self.at_layer[0]:
            self.at0 = at_model(**at_kws[0])
        if self.at_layer[1]:
            self.at1 = at_model(**at_kws[1])
        if self.at_layer[2]:
            self.at2 = at_model(**at_kws[2]) 
        if self.at_layer[3]:
            self.at3 = at_model(**at_kws[3])
        if self.at_layer[4]:
            self.at4 = at_model(**at_kws[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def  forward(self, x):
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        if self.low_head_type:
            x = self.low_head_module(x)
        else:
            x = self.maxpool(x)  # 56

        if self.at_layer[0]:
            x = self.at0(x)

        x = self.layer1(x)  # 56
        if self.at_layer[1]:
            x = self.at1(x)

        x = self.layer2(x)  # 28
        if self.at_layer[2]:
            x = self.at2(x)

        x = self.layer3(x)  # 14
        if self.at_layer[3]:
            x = self.at3(x)

        x = self.layer4(x)  # 7
        if self.at_layer[4]:
            x = self.at4(x)

        if self._bc_kw:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(self.drop(x))
        return x


def resnet_at(in_channels, num_classes=7, weight_path="", **kw):
    filters = kw['filters'] if 'filters' in kw else [64, 64, 128, 256, 512]
    at_type = kw['at_type'] if 'at_type' in kw else 'CB'
    at_kws = kw['at_kws'] if 'at_kws' in kw else [
        {
            'inplanes': filters[i]
        } for i in range(len(filters))
    ]
    at_layer = kw['at_layer'] if 'at_layer' in kw else [0, 1, 0, 0, 0]
    resnet_type = kw['resnet_type'] if 'resnet_type' in kw else 'resnet18'
    pretrained = kw['pretrained'] if 'pretrained' in kw else False
    drop = kw['drop'] if 'drop' in kw else 0.0
    bc_kw = kw['bc_kw'] if 'bc_kw' in kw else None
    low_head_type = kw['low_head_type'] if 'low_head_type' in kw else None
    low_head_kw = kw['low_head_kw'] if 'low_head_kw' in kw else None
    model = ResNetAT(at_type=at_type, at_kws=at_kws, low_head_type=low_head_type, 
                    low_head_kw=low_head_kw, bc_kw=bc_kw, at_layer=at_layer, 
                    resnet_type=resnet_type, pretrained=pretrained,
                    in_channels=in_channels, num_classes=num_classes, drop=drop)
    return model
    