import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    '''
    Simpfy implementation, refer to https://blog.csdn.net/weixin_41693877/article/details/108395270
    '''
    def __init__(self, in_channnels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channnels_x, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())

    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = F.interpolate(self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.psi(F.relu(x1 + g1, inplace=True))
        return out * x


class DownsampleLayer(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out = self.downsample(out)
        return out


class AttentionGate(nn.Module):

    def __init__(self, in_channels, ratio=0.5):
        super(AttentionGate, self).__init__()
        self.down = DownsampleLayer(in_channels, in_channels * 2)
        self.attention = AttentionBlock(in_channels, in_channels * 2, int(in_channels * ratio))
        self.bn = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        g = self.down(x)
        return F.relu(x + self.bn(self.attention(x, g)), inplace=True)


class AttentionGate2(nn.Module):

    def __init__(self, in_channels, ratio=0.5):
        super(AttentionGate2, self).__init__()
        self.down1 = DownsampleLayer(in_channels, in_channels * 2)
        self.down2 = DownsampleLayer(in_channels * 2, in_channels * 4)
        self.attention1 = AttentionBlock(in_channels, in_channels * 2, int(in_channels * ratio))
        self.attention2 = AttentionBlock(in_channels * 2, in_channels * 4, int(in_channels * 2 * ratio))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        g1 = F.relu(x1 + self.bn2(self.attention2(x1, x2)), inplace=True)
        g2 = F.relu(x + self.bn1(self.attention1(x, g1)), inplace=True)
        return g2


if __name__ == "__main__":
    print('test')
    x = torch.rand(size=(4, 64, 48, 48))
    model = AttentionGate(in_channels=64)
    y = model(x)
    print(x.shape, y.shape)
    model = AttentionGate2(in_channels=64)
    y = model(x)
    print(model)
    print(x.shape, y.shape)
