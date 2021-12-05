import torch.nn as nn
from .lightcnn import LightCNN, LowHead
from .attention_gate import AttentionGate


class LightCNNAG(LightCNN):

    def __init__(self, low_head, filters=[64, 128, 256, 512, 1024], num_classes=7, ag_layer=[1, 1, 1, 1, 1]):
        super(LightCNNAG, self).__init__(low_head, filters, num_classes)
        self.ag_layer = ag_layer
        if self.ag_layer[0]:
            self.ag0 = AttentionGate(filters[0])
        if self.ag_layer[1]:
            self.ag1 = AttentionGate(filters[1])
        if self.ag_layer[2]:
            self.ag2 = AttentionGate(filters[2]) 
        if self.ag_layer[3]:
            self.ag3 = AttentionGate(filters[3])
        if self.ag_layer[4]:
            self.ag4 = AttentionGate(filters[4])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
    
    def forward(self, x):

        x = self.low_head(x)
        if self.ag_layer[0]:
            x = self.ag0(x)
        x = self.layer1(x)
        if self.ag_layer[1]:
            x = self.ag1(x)
        x = self.layer2(x)
        if self.ag_layer[2]:
            x = self.ag2(x)
        x = self.layer3(x)
        if self.ag_layer[3]:
            x = self.ag3(x)
        x = self.layer4(x)
        if self.ag_layer[4]:
            x = self.ag4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 


def make_lightcnn_ag(in_channels, num_classes=7, weight_path="", 
                    filters=[64, 128, 256, 518, 1024], ag_layer=[0, 0, 1, 1, 1]):
    filters = filters
    low_head = LowHead(in_channels, filters[0])
    ag_layer = ag_layer
    model = LightCNNAG(low_head=low_head, filters=filters, num_classes=num_classes, ag_layer=ag_layer)
    return model


def lightcnn_ag(in_channels, num_classes=7, weight_path="", **kw):
    filters = kw['filters'] if 'filters' in kw else [64, 128, 256, 518, 1024]
    ag_layer = kw['ag_layer'] if 'ag_layer' in kw else [1, 1, 1, 1, 1] 
    return make_lightcnn_ag(in_channels=in_channels, num_classes=num_classes, weight_path=weight_path,
                            filters=filters, ag_layer=ag_layer)