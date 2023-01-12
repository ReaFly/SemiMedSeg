import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .module import GFF


def cat(x1, x2, x3=None, dim=1):
    if x3 == None:
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        return x
    else:
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        diffY = torch.tensor([x.size()[2] - x3.size()[2]])
        diffX = torch.tensor([x.size()[3] - x3.size()[3]])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        if transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4,
                                   in_channels // 4,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load("../../backbone/resnet34.pth"))

        if in_channels == 3:
            self.encoder1_conv = resnet.conv1
        else:
            self.encoder1_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_maxpool = self.maxpool(e1)

        e2 = self.encoder2(e1_maxpool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5


class MyModel(nn.Module):
    def __init__(self,
                 num_classes=1,
                 in_channels=3
                 ):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        
        # seg-Decoder
        self.segDecoder5 = DecoderBlock(512, 512)
        
        self.segDecoder4 = DecoderBlock(512 + 256, 256)
        
        self.segDecoder3 = DecoderBlock(256 + 128, 128)
       
        self.segDecoder2 = DecoderBlock(128 + 64, 64)
        
        self.segDecoder1 = DecoderBlock(64 + 64, 64)

        self.segSideout5 = SideoutBlock(512, 1)
        self.segSideout4 = SideoutBlock(256, 1)
        self.segSideout3 = SideoutBlock(128, 1)
        self.segSideout2 = SideoutBlock(64, 1)

        self.segconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        # inpaint-Decoder
        
        self.inpDecoder5 = DecoderBlock(512, 512, transpose=True)
        self.inpDecoder4 = DecoderBlock(512 + 256, 256, transpose=True)
        self.inpDecoder3 = DecoderBlock(256 + 128, 128, transpose=True)
        self.inpDecoder2 = DecoderBlock(128 + 64, 64, transpose=True)
        self.inpDecoder1 = DecoderBlock(64 + 64, 64, transpose=True)

        self.inpSideout5 = SideoutBlock(512, 3)
        self.inpSideout4 = SideoutBlock(256, 3)
        self.inpSideout3 = SideoutBlock(128, 3)
        self.inpSideout2 = SideoutBlock(64, 3)

        self.gate1 = GFF(64)
        self.gate2 = GFF(64)
        self.gate3 = GFF(128)
        self.gate4 = GFF(256)
        self.gate5 = GFF(512)

        self.inpconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(32, in_channels, 1))

    def forward(self, x):
        ori = x
        bs, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        """Seg-branch"""
        e1, e2, e3, e4, e5 = self.encoder(x)
        d5 = self.segDecoder5(e5)
        d4 = self.segDecoder4(cat(d5, e4))
        d3 = self.segDecoder3(cat(d4, e3))
        d2 = self.segDecoder2(cat(d3, e2))
        d1 = self.segDecoder1(cat(d2, e1))
        
        mask = self.segconv(d1)
        mask = torch.sigmoid(mask)
        #sidecopy = sidemask.detach().clone()
        mask_binary = (mask > 0.5).float()
        mask_rbinary = (mask < 0.5).float()
        cut_ori = ori * mask_rbinary
        inpe1, inpe2, inpe3, inpe4, inpe5 = self.encoder(cut_ori)
        e1, e2, e3, e4, e5 = self.encoder(x)

        ge1 = self.gate1(e1, inpe1)
        ge2 = self.gate2(e2, inpe2)
        ge3 = self.gate3(e3, inpe3)
        ge4 = self.gate4(e4, inpe4)
        ge5 = self.gate5(e5, inpe5)
        
        d5 = self.segDecoder5(ge5)
        sidemask5 = self.segSideout5(d5)
        sidemask5 = torch.sigmoid(sidemask5)
        
        d4 = self.segDecoder4(cat(d5, ge4))
        sidemask4 = self.segSideout4(d4)
        sidemask4 = torch.sigmoid(sidemask4)
       
        d3 = self.segDecoder3(cat(d4, ge3))
        sidemask3 = self.segSideout3(d3)
        sidemask3 = torch.sigmoid(sidemask3)
        
        d2 = self.segDecoder2(cat(d3, ge2))
        sidemask2 = self.segSideout2(d2)
        sidemask2 = torch.sigmoid(sidemask2)
        
        d1 = self.segDecoder1(cat(d2, ge1))
        
        mask = self.segconv(d1)
        #maskh, maskw = mask.shape[2], mask.shape[3]
        #if maskh != H or maskw != W:
        #   mask = F.interpolate(mask, size=(H, W), mode='bilinear')
        mask = torch.sigmoid(mask)

        
        inpd5 = self.inpDecoder5(ge5)
        inpimg5 = self.inpSideout5(inpd5)
        inpd4 = self.inpDecoder4(cat(inpd5, ge4))
        inpimg4 = self.inpSideout4(inpd4)
        inpd3 = self.inpDecoder3(cat(inpd4, ge3))
        inpimg3 = self.inpSideout3(inpd3)
        inpd2 = self.inpDecoder2(cat(inpd3, ge2))
        inpimg2 = self.inpSideout2(inpd2)
        inpd1 = self.inpDecoder1(cat(inpd2, ge1))
        inpimg = self.inpconv(inpd1)
        mask_binary = (mask > 0.5).float()
        mask_rbinary = (mask < 0.5).float()
        inpimg = inpimg * mask_binary + ori * mask_rbinary
        
        return mask, sidemask2, sidemask3, sidemask4, sidemask5, inpimg, inpimg2, inpimg3, inpimg4, inpimg5

