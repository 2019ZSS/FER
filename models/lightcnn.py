import torch
import torch.nn as nn
import torch.nn.functional as F


def depthwise_conv(inp, outp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
 
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class DepthWiseConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(DepthWiseConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    groups=in_ch)

        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


def get_conv_layer(conv, in_ch, out_ch, kernel_size, stride):
    return nn.Sequential(
        conv(in_ch, out_ch, kernel_size, stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.conv1 = get_conv_layer(nn.Conv2d, in_ch, out_ch, 1, 1)
        self.conv2 = get_conv_layer(DepthWiseConv, out_ch, out_ch, 3, 1)
        self.conv3 = get_conv_layer(nn.Conv2d, out_ch, out_ch, 1, 1)
        self.conv4 = get_conv_layer(nn.Conv2d, in_ch, out_ch, 1, 2)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
    
    def forward(self, x):
        y = self.conv4(x)
        x = self.pool(self.conv3(self.conv2(self.conv1(x))))
        x = x + y 
        return x 


class LowHead(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0):
        super(LowHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, 
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act1(self.bn1(self.conv1(x)))


class LightCNN(nn.Module):

    def __init__(self, low_head, filters=[64, 128, 256, 512, 1024], num_classes=7):
        super(LightCNN, self).__init__()
        self.low_head = low_head
        self.layer1 = BasicBlock(filters[0], filters[1])
        self.layer2 = BasicBlock(filters[1], filters[2])
        self.layer3 = BasicBlock(filters[2], filters[3])
        self.layer4 = BasicBlock(filters[3], filters[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        
        x = self.low_head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def lightcnn(in_channels, num_classes=7, weight_path=""):
    filters = [64, 128, 256, 518, 1024]
    low_head = LowHead(in_channels, filters[0])
    model = LightCNN(low_head=low_head, filters=filters, num_classes=num_classes)
    return model

def lightcnn_dropout1(in_channels, num_classes=7, weight_path=""):
    filters = [64, 128, 256, 518, 1024]
    low_head = LowHead(in_channels, filters[0])
    model = LightCNN(low_head=low_head, filters=filters, num_classes=num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(filters[-1], num_classes)
    )
    return model


if __name__ == "__main__":
    print('test')
    in_ch = 1
    x = torch.rand(size=(4, in_ch, 48, 48))
    model = lightcnn(in_channels=in_ch, num_classes=7)
    y = model(x)
    print(x.shape, y.shape)
   