import torch.nn as nn
from .lightcnn import LightCNN, LowHead
from .at import (
    ContextBlock,
    StripPooling,
)


def getATModule(at_type):
    if at_type == 'CB':
        return ContextBlock
    if at_type == 'SP':
        return StripPooling 
    raise NotImplementedError('{} not implemented'.format(at_type))


class LightCNNAT(LightCNN):

    def __init__(self, low_head, at_type, at_kws, filters=[64, 128, 256, 512, 1024], num_classes=7, at_layer=[0, 0, 0, 0, 1]):
        super(LightCNNAT, self).__init__(low_head, filters, num_classes)
        at_model = getATModule(at_type)
        self.at_layer = at_layer
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

    def forward(self, x):

        x = self.low_head(x)
        if self.at_layer[0]:
            x = self.at0(x)
        x = self.layer1(x)
        if self.at_layer[1]:
            x = self.at1(x)
        x = self.layer2(x)
        if self.at_layer[2]:
            x = self.at2(x)
        x = self.layer3(x)
        if self.at_layer[3]:
            x = self.at3(x)
        x = self.layer4(x)
        if self.at_layer[4]:
            x = self.at4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def make_lightcnn_at(in_channels, at_type, at_kws, num_classes=7, weight_path="", 
                    filters=[64, 128, 256, 518, 1024], at_layer=[0, 0, 0, 0, 1], drop=0.):
    filters = filters
    low_head = LowHead(in_channels, filters[0])
    at_layer = at_layer
    model = LightCNNAT(low_head=low_head, at_type=at_type, at_kws=at_kws, filters=filters, num_classes=num_classes, at_layer=at_layer)
    if drop > 0:
        model.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(filters[-1], num_classes)
        )
    return model


def lightcnn_at(in_channels, num_classes=7, weight_path="", **kw):
    filters = kw['filters'] if 'filters' in kw else [64, 128, 256, 518, 1024]
    at_layer = kw['at_layer'] if 'at_layer' in kw else [0, 0, 0, 0, 1] 
    drop = kw['drop'] if 'drop' in kw else 0.0
    at_type = kw['at_type'] if 'at_type' in kw else 'CB'
    at_kws = kw['at_kws'] if 'at_kws' in kw else [
        {
            'inplanes': filters[i]
        } for i in range(len(filters))
    ]
    return make_lightcnn_at(in_channels=in_channels, at_type=at_type, at_kws=at_kws,
                            num_classes=num_classes, weight_path=weight_path,
                            filters=filters, at_layer=at_layer, drop=drop)