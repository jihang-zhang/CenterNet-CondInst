import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcn import DeformConvPack, ModulatedDeformConvPack

class DCN3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type='DCNv2', upsample=False):
        super().__init__()

        if conv_type == 'DCN':
            Conv2d = DeformConvPack
        elif conv_type == 'DCNv2':
            Conv2d = ModulatedDeformConvPack
        else:
            raise NotImplementedError('Unknown DCN type {}; please select from ["DCN", "DCNv2"]'.format(conv_type))

        self.upsample = upsample
        self.block = nn.Sequential(
            Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, dilation=1, deformable_groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class TTFDecoder(nn.Module):
    def __init__(
        self, 
        inplanes=(64, 128, 256, 512),
        planes=(256, 128, 64),
        shortcut_kernel=3,
        shortcut_cfg=(1, 2, 3),
        dcn_type='DCNv2'):
        super(TTFDecoder, self).__init__()

        assert len(planes) in [2, 3, 4]
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)
        self.planes = planes
        self.dcn_type = dcn_type

        self.deconv_layers = nn.ModuleList([
            self.build_upsample(inplanes[-1], planes[0]),
            self.build_upsample(planes[0], planes[1])
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self.build_upsample(planes[i - 1], planes[i]))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self.build_shortcut(
            inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
            kernel_size=shortcut_kernel, padding=padding)

    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes):
        return DCN3x3BNReLU(inplanes, planes, conv_type=self.dcn_type, upsample=True)


    def forward(self, feats):
        feats = feats[-4:]
        x = feats[-1]
        for i, upsample_layer in enumerate(self.deconv_layers):
            x = upsample_layer(x)
            if i < len(self.shortcut_layers):
                shortcut = self.shortcut_layers[i](feats[-i - 2])
                x = x + shortcut

        return x
