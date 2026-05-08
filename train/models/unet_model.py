from typing import List
import torch
import torch.nn as nn

def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

def _up_block(cin: int, skip_ch: int, cout: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin + skip_ch, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        ch1, ch2, ch3, ch4, ch5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        self.stem = _conv_block(in_ch, ch1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = _conv_block(ch1, ch2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = _conv_block(ch2, ch3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = _conv_block(ch3, ch4)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = _conv_block(ch4, ch5)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.stem(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        return [x2, x3, x4, x5]

class UNetDecoder(nn.Module):
    def __init__(self, chs: List[int], out_ch: int = 1):
        super().__init__()
        c1_4, c1_8, c1_16, c1_32 = chs

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _up_block(c1_32, c1_16, c1_16)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _up_block(c1_16, c1_8, c1_8)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _up_block(c1_8, c1_4, c1_4)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(c1_4, out_ch, kernel_size=1)

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        x1_4, x1_8, x1_16, x1_32 = skips

        y = self.up1(x1_32)
        y = self.dec1(torch.cat([y, x1_16], dim=1))

        y = self.up2(y)
        y = self.dec2(torch.cat([y, x1_8], dim=1))

        y = self.up3(y)
        y = self.dec3(torch.cat([y, x1_4], dim=1))

        y = self.up4(y)
        logits = self.head(y)
        return logits

class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        chs = [base_ch*2, base_ch*4, base_ch*8, base_ch*16]
        self.decoder = UNetDecoder(chs, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        logits = self.decoder(skips)
        return logits

if __name__ == "__main__":
    net = UNet(in_ch=3, out_ch=1, base_ch=32)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Model out:", y.shape)
    tot_params = sum(p.numel() for p in net.parameters())
    print(f"Total params: {tot_params/1e6:.3f}M")
