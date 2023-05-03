import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class FcnNetMep(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(FcnNetMep, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.modules.spatial_ocr_block_mep import FCN_Module_Mep
        self.fcn_head = FCN_Module_Mep(
            configer=configer,
            features=2048,
            hidden_features=256,
            out_features=256,
            num_classes=self.num_classes,
            bn_type=self.configer.get("network", "bn_type"),
        )

        self.head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        # mep
        x = self.fcn_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x



class ASPOCRNetMep(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(ASPOCRNetMep, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.modules.spatial_ocr_block_mep import SpatialOCR_ASP_Module_Mep
        self.asp_ocr_head = SpatialOCR_ASP_Module_Mep(
            configer=configer,
            features=2048,
            hidden_features=256,
            out_features=256,
            num_classes=self.num_classes,
            bn_type=self.configer.get("network", "bn_type"),
        )

        self.head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        # mep
        x, proj = self.asp_ocr_head(x[-1], x_dsn)
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj
