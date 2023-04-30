##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class HRNet_W48_DC(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_DC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                512, bn_type=self.configer.get("network", "bn_type")
            ),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout2d(0.10),
            nn.Conv2d(
                512,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        # contrast
        from lib.models.modules.contrast import Contrast_Module
        self.contrast_head = Contrast_Module(configer)

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        feats = self.seg_head(feats)

        # contrast
        bk = [feat1, feat2, feat3, feat4]
        output, feats = self.contrast_head(bk, feats)     
        # contrast

        out = self.cls_head(feats)
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out, output


class HRNet_W48_ASPOCR_Mep(nn.Module):
    def __init__(self, configer):
        super(HRNet_W48_ASPOCR_Mep, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384

        # Mep
        from lib.models.modules.spatial_ocr_block_mep import SpatialOCR_ASP_Module_Mep
        self.asp_ocr_head = SpatialOCR_ASP_Module_Mep(
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


class HRNet_W48_OCR_DC(nn.Module):
    def __init__(self, configer):
        super(HRNet_W48_OCR_DC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module

        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module

        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get("network", "bn_type")
            ),
            nn.Conv2d(
                in_channels,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        # contrast
        from lib.models.modules.contrast import Contrast_Module
        self.contrast_head = Contrast_Module(configer)

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        # contrast
        bk = [feat1, feat2, feat3, feat4]
        output, feats = self.contrast_head(bk, feats)     
        # contrast

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out, output


class HRNet_W48_OCR_B(nn.Module):
    """
    Considering that the 3x3 convolution on the 4x resolution feature map is expensive,
    we can decrease the intermediate channels from 512 to 256 w/o performance loss.
    """

    def __init__(self, configer):
        super(HRNet_W48_OCR_B, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720  # 48 + 96 + 192 + 384
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get("network", "bn_type")),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module

        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module

        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=256,
            key_channels=128,
            out_channels=256,
            scale=1,
            dropout=0.05,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get("network", "bn_type")),
            nn.Conv2d(
                256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
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

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return out_aux, out