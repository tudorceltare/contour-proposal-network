import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # the bottom part of the UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for idx, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # There must be a better way to do this with zip
        # for skip_connection, up in zip(skip_connections, self.ups):
        #     x = up(x)
        #     concat_skip = torch.cat((skip_connection, x), 1)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), 1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class ResUNET(nn.Module):
    def __init__(self, encoder, out_channels=1, freeze_encoder=True):
        super(ResUNET, self).__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = not freeze_encoder

        # assuming that the encoder is a resnet18
        self.encoder_layers = list(encoder.children())
        self.layer_0 = nn.Sequential(*self.encoder_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer_0_1x1 = ConvReLu(64, 64, kernel_size=1, stride=1, padding=0)
        self.layer_1 = nn.Sequential(*self.encoder_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer_1_1x1 = ConvReLu(64, 64, kernel_size=1, stride=1, padding=0)
        self.layer_2 = self.encoder_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_2_1x1 = ConvReLu(128, 128, kernel_size=1, stride=1, padding=0)
        self.layer_3 = self.encoder_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_3_1x1 = ConvReLu(256, 256, kernel_size=1, stride=1, padding=0)
        self.layer_4 = self.encoder_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer_4_1x1 = ConvReLu(512, 512, kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = ConvReLu(256 + 512, 512, kernel_size=3, padding=1)
        self.conv_up2 = ConvReLu(128 + 512, 256, kernel_size=3, padding=1)
        self.conv_up1 = ConvReLu(64 + 256, 256, kernel_size=3, padding=1)
        self.conv_up0 = ConvReLu(64 + 256, 128, kernel_size=3, padding=1)

        self.conv_original_size0 = ConvReLu(3, 64, kernel_size=3, padding=1)
        self.conv_original_size1 = ConvReLu(64, 64, kernel_size=3, padding=1)
        self.conv_original_size2 = ConvReLu(64 + 128, 64, kernel_size=3, padding=1)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer_0(x)
        layer1 = self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)

        layer4 = self.layer_4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer_3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer_2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer_1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer_0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
