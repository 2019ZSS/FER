import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):

    def __init__(self, in_channels: int):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = in_channels // 2
        self.conv_phi = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel, 
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel, 
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel, 
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=in_channels, 
                                    kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        # [B, C, H, W]
        b, c, h, w = x.size()
        # [B, C / 2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [B, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [B, H * W, H * W]
        mul_theta_phi = F.softmax(torch.matmul(x_theta, x_phi), dim=1)
        # [B, H*W, C / 2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [B, C / 2, H, w]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b, self.inter_channel, h, w)
        # [B, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out
 

class ContextBlock(nn.Module):
    '''
    Global Context Network（GCNet）
    Paper:  Global Context Network（GCNet）
    Code: https://github.com/xvjiarui/GCNet
    Analysis: https://blog.csdn.net/sinat_17456165/article/details/106760606
    '''
    def __init__(self, inplanes, ratio=1.0/16, pooling_type='att', fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']
 
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
 
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
 
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
 
 
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context
 
    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, up_kwargs, norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class BilinearCNN(nn.Module):

    def __init__(self, in_channels=512, inter_channels=512, num_classes=7, drop=0., eps=1e-10):
        super(BilinearCNN, self).__init__()
        self._in_channels = in_channels
        self._inter_channels = inter_channels
        self._eps = eps
        if self._in_channels != self._inter_channels:
            self.conv = nn.Sequential(nn.Conv2d(in_channels=self._in_channels, out_channels=self._inter_channels, kernel_size=1), 
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(True))
        self.fc = nn.Linear(in_features=inter_channels**2, out_features=num_classes) 
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        batch_size, in_channels, h, w = x.size()
        feature_size = h * w 
        if self._in_channels != self._inter_channels:
            x = self.conv(x)
        x = x.view(batch_size, self._inter_channels, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + self._eps))
        return self.fc(self.drop(x))


if __name__=='__main__':
    model = NonLocalBlock(in_channels=16)
    print(model)
    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)

    in_tensor = torch.ones((12, 64, 128, 128))
    cb = ContextBlock(inplanes=64, ratio=1./16.,pooling_type='att')
    out_tensor = cb(in_tensor)
    print(in_tensor.shape)
    print(out_tensor.shape)

    in_channels = 64
    pool_size = (3, 3)
    norm_layer = nn.BatchNorm2d
    up_kwargs = {
        'scale_factor': None, 
        'mode': 'bilinear', 
        'align_corners': False
    }
    x = torch.rand(size=(4, in_channels, 16, 16))
    sp = StripPooling(in_channels, pool_size, up_kwargs, norm_layer)
    y = sp(x)
    print(x.shape, y.shape)

    in_channels = 512
    inter_channels = 512
    x = torch.randn(size=(4, in_channels, 16, 16))
    model = BilinearCNN(in_channels=in_channels, inter_channels=inter_channels, drop=0.5)
    y = model(x)
    print(x.shape, y.shape)