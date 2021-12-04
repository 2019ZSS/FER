import torch
from models import lightcnn_ag


if __name__ == "__main__":
    in_channels = 1
    x = torch.rand(size=(4, in_channels, 48, 48))
    model = lightcnn_ag(in_channels)
    y = model(x)
    print(x.shape, y.shape)