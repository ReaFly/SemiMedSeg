import torch
import torch.nn as nn


class GFF(nn.Module):

    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()

        self.reset_gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=kernel_size, padding=padding)
        self.select_gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=kernel_size, padding=padding)
        self.outconv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=kernel_size, padding=padding)
        self.activate = nn.ReLU(inplace=True)
        
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.select_gate.weight)
        nn.init.orthogonal_(self.outconv.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.select_gate.bias, 0.)
        nn.init.constant_(self.outconv.bias, 0.)

    def forward(self, seg_feat, inp_feat):

        concat_inputs = torch.cat([seg_feat, inp_feat], dim=1)
        reset = torch.sigmoid(self.reset_gate(concat_inputs))
        select = torch.sigmoid(self.select_gate(concat_inputs))
        out_inputs = self.activate(self.outconv(torch.cat([seg_feat, inp_feat * reset], dim=1)))
        out_final = seg_feat * (1 - select) + out_inputs * select

        return out_final




