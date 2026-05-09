import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CannyNet(nn.Module):
    def __init__(self, threshold=3.0, use_cuda=True):
        super(CannyNet, self).__init__()
        self.threshold = threshold
        self.use_cuda = use_cuda
        # Simplified Sobel-based edge detection for self-sufficiency
        self.filter_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.filter_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.filter_x.weight.data = sobel_x
        self.filter_y.weight.data = sobel_y
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        dx = self.filter_x(gray)
        dy = self.filter_y(gray)
        mag = torch.sqrt(dx**2 + dy**2 + 1e-6)
        return None, None, None, None, mag, None

class EFBranch(nn.Module):
    """
    Simplified Edge-Feedback Branch
    """
    def __init__(self, in_ch=32):
        super(EFBranch, self).__init__()
        self.conv1 = BasicConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = BasicConv2d(in_ch, in_ch, 3, padding=1)
        self.edge_conv = nn.Conv2d(1, in_ch, 1)

    def forward(self, x, edge):
        # x: features, edge: edge map
        edge = F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge_feat = self.edge_conv(edge)
        out = x + edge_feat
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class CDPolypNet(nn.Module):
    def __init__(self, model_type="resnet", out_ch=1, use_pretrained=False):
        super().__init__()
        # Simplified backbone for scratch training
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        
        self.canny = CannyNet()
        self.efb = EFBranch(in_ch=256)
        
        self.decoder = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        # Edge Detection
        _, _, _, _, edge, _ = self.canny(x)
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Edge Feedback
        feat = self.efb(x3, edge)
        
        # Decoder
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        
        return logits

def get_cd_polypnet(use_pretrained=False, **kwargs):
    out_ch = kwargs.get("out_ch", 1)
    return CDPolypNet(out_ch=out_ch, use_pretrained=use_pretrained)
