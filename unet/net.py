import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None) -> None:
        if mid_c is None:
            mid_c = out_c
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class CropAndConcat(nn.Module):
    def __init__(self) -> None:
        super(CropAndConcat, self).__init__()

    def forward(self, x1, x2):
        x1_size = x1.size()[2]
        x2_size = x2.size()[2]

        half_diff = (x1_size - x2_size) // 2
        x1_cropped = x1[
                     :, :, half_diff: (x1_size - half_diff), half_diff: (x1_size - half_diff)
                     ]  # crop
        return torch.cat([x1_cropped, x2], dim=1)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1) -> None:
        super(UNet, self).__init__()
        self.max_pool_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(in_channels, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)
        self.down_conv_5 = DoubleConv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = DoubleConv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = DoubleConv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = DoubleConv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = DoubleConv(128, 64)

        self.crop_and_concat_5 = CropAndConcat()
        self.crop_and_concat_4 = CropAndConcat()
        self.crop_and_concat_3 = CropAndConcat()
        self.crop_and_concat_2 = CropAndConcat()
        self.crop_and_concat_1 = CropAndConcat()

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x1_1 = self.down_conv_1(x)
        x1_2 = self.max_pool_2d(x1_1)

        x2_1 = self.down_conv_2(x1_2)
        x2_2 = self.max_pool_2d(x2_1)

        x3_1 = self.down_conv_3(x2_2)
        x3_2 = self.max_pool_2d(x3_1)

        x4_1 = self.down_conv_4(x3_2)
        x4_2 = self.max_pool_2d(x4_1)

        x5 = self.down_conv_5(x4_2)

        # decoder
        x6 = self.up_trans_1(x5)
        x6 = self.crop_and_concat_5(x4_1, x6)
        x6 = self.up_conv_1(x6)

        x7 = self.up_trans_2(x6)
        x7 = self.crop_and_concat_4(x3_1, x7)
        x7 = self.up_conv_2(x7)

        x8 = self.up_trans_3(x7)
        x8 = self.crop_and_concat_3(x2_1, x8)
        x8 = self.up_conv_3(x8)

        x9 = self.up_trans_4(x8)
        x9 = self.crop_and_concat_2(x1_1, x9)
        x9 = self.up_conv_4(x9)

        output = self.final_conv(x9)

        return output


if __name__ == "__main__":
    img = torch.randn((1, 3, 512, 512))
    print("input image shape:")
    print(img.shape)
    model = UNet()
    print("output image shape:")
    print(model(img).shape)
