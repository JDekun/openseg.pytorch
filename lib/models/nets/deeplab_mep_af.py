import torch.nn as nn
import torch
from torch import nn

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module based on DeepLab v3 settings"""

    def __init__(self,configer, in_dim, out_dim, d_rate=[12, 24, 36], bn_type=None):
        super(ASPPModule, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1,
                                          bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[0],
                                          dilation=d_rate[0], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[1],
                                          dilation=d_rate[1], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[2],
                                          dilation=d_rate[2], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b4 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=1, bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_dim, out_dim, kernel_size=1, padding=0,
                      bias=False),
            ModuleHelper.BNReLU(out_dim, bn_type=bn_type),
            nn.Dropout2d(0.1)
        )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, out_dim)

    def forward(self, x):
        h, w = x.size()[2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        out = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)
        output = self.project(out)

        # mep
        feat = [output]
        proj = self.mep_head(feat)     
        # mep
        return output, proj


class DeepLabHead(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, configer, num_classes, bn_type=None):
        super(DeepLabHead, self).__init__()
        # auxiliary loss
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3,
                                                 stride=1, padding=1),
                                       ModuleHelper.BNReLU(512, bn_type=bn_type),
                                       nn.Dropout2d(0.1),
                                       nn.Conv2d(512, num_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True))
        # main pipeline
        self.layer_aspp = ASPPModule(configer, 2048, 256, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1,
                                    stride=1, bias=True))

    def forward(self, x):
        # auxiliary supervision
        x_dsn = self.layer_dsn(x[2])
        # aspp module
        x_aspp, proj = self.layer_aspp(x[3])
        # refine module
        x_seg = self.refine(x_aspp)

        return [x_seg, x_dsn, proj]


class DeepLabV3_MEP_AF(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3_MEP_AF, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead(configer, num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        x = self.backbone(x_)

        x = self.decoder(x[-4:])

        return x[1], x[0], x[2]
