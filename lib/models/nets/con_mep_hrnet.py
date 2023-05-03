import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class HRNet_W48_ASPOCR_MLM(nn.Module):
    def __init__(self, configer):
        super(HRNet_W48_ASPOCR_MLM, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384

        # Mep
        from lib.models.modules.spatial_ocr_block_mep import SpatialOCR_ASP_Module_MLM
        self.asp_ocr_head = SpatialOCR_ASP_Module_MLM(
            configer=configer,
            features=720,
            hidden_features=256,
            out_features=256,
            dilations=(24, 48, 72),
            num_classes=self.num_classes,
            bn_type=self.configer.get("network", "bn_type"),
        )

        self.cls_head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        # mep
        feats, proj = self.asp_ocr_head(feats, out_aux)
        # mep

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out, proj