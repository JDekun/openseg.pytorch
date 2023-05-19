import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper

class RES_DEEPLABV3_ASP_MEP_IN(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_DEEPLABV3_ASP_MEP_IN, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import DEEPLABV3_ASP_MEP_IN
        self.deeplabv3_asp_head = DEEPLABV3_ASP_MEP_IN(
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
        x, proj = self.deeplabv3_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_DEEPLABV3_ASP_MEP_AF(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_DEEPLABV3_ASP_MEP_AF, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import DEEPLABV3_ASP_MEP_AF
        self.deeplabv3_asp_head = DEEPLABV3_ASP_MEP_AF(
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
        x, proj = self.deeplabv3_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_DEEPLABV3_ASP_MEP_BE(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_DEEPLABV3_ASP_MEP_BE, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import DEEPLABV3_ASP_MEP_BE
        self.deeplabv3_asp_head = DEEPLABV3_ASP_MEP_BE(
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
        x, proj = self.deeplabv3_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj


class RES_DEEPLABV3_ASP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_DEEPLABV3_ASP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import DEEPLABV3_ASP
        self.deeplabv3_asp_head = DEEPLABV3_ASP(
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
        x = self.deeplabv3_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x


class RES_OCR_ASP_0_MEP_BE(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_OCR_ASP_0_MEP_BE, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import OCR_ASP_0_MEP_BE
        self.ocr_asp_head = OCR_ASP_0_MEP_BE(
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
        x, proj = self.ocr_asp_head(x[-1], x_dsn)
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_OCR_ASP_0_MEP_AF(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_OCR_ASP_0_MEP_AF, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import OCR_ASP_0_MEP_AF
        self.ocr_asp_head = OCR_ASP_0_MEP_AF(
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
        x, proj = self.ocr_asp_head(x[-1], x_dsn)
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_OCR_ASP_0_MEP_IN(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_OCR_ASP_0_MEP_IN, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import OCR_ASP_0_MEP_IN
        self.ocr_asp_head = OCR_ASP_0_MEP_IN(
            configer=configer,
            features=2048,
            hidden_features=512,
            out_features=512,
            num_classes=self.num_classes,
            bn_type=self.configer.get("network", "bn_type"),
        )

        self.head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
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
        x, proj = self.ocr_asp_head(x[-1], x_dsn)
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj
    
class RES_OCR_ASP_0(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_OCR_ASP_0, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import OCR_ASP_0
        self.ocr_asp_head = OCR_ASP_0(
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
        x = self.ocr_asp_head(x[-1], x_dsn)
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x

class RES_FCN_ASP_0(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_0, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_0
        self.fcn_asp_head = FCN_ASP_0(
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
        x = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x

class RES_FCN_ASP_0_MEP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_0_MEP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_0_Mep
        self.fcn_asp_head = FCN_ASP_0_Mep(
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
        x, proj = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_FCN_ASP_1_MEP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_1_MEP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_1_Mep
        self.fcn_asp_head = FCN_ASP_1_Mep(
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
        x, proj = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_FCN_ASP_2_MEP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_2_MEP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_2_Mep
        self.fcn_asp_head = FCN_ASP_2_Mep(
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
        x, proj = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_FCN_ASP_4_MEP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_4_MEP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_4_Mep
        self.fcn_asp_head = FCN_ASP_4_Mep(
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
        x, proj = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_FCN_ASP_3_MEP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_3_MEP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_3_Mep
        self.fcn_asp_head = FCN_ASP_3_Mep(
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
        x, proj = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj


class RES_FCN512_ASP_3_MEP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN512_ASP_3_MEP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN512_ASP_3_Mep
        self.fcn_asp_head = FCN512_ASP_3_Mep(
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
        x, proj = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x, proj

class RES_FCN_ASP_3(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP_3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP_3
        self.fcn_asp_head = FCN_ASP_3(
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
        x = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x

class RES_FCN_ASP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_FCN_ASP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import FCN_ASP
        self.fcn_asp_head = FCN_ASP(
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
        x = self.fcn_asp_head(x[-1])
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x



class RES_OCR_ASP(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(RES_OCR_ASP, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get("network", "backbone"):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # Mep
        from lib.models.nets.contrast_asp import OCR_ASP
        self.ocr_asp_head = OCR_ASP(
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
        x = self.ocr_asp_head(x[-1], x_dsn)
        # mep
        x = self.head(x)
        x_dsn = F.interpolate(
            x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        x = F.interpolate(
            x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )
        return x_dsn, x
