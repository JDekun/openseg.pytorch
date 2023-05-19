import os
import pdb
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper

from lib.models.modules.spatial_ocr_block import SpatialGather_Module

class DEEPLABV3_ASP_MEP_IN(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(DEEPLABV3_ASP_MEP_IN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        
        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features * 5)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.conv1(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)

        # mep
        feat = [out]
        proj = self.mep_head(feat)     
        # mep
        
        return output, proj
    
class DEEPLABV3_ASP_MEP_AF(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(DEEPLABV3_ASP_MEP_AF, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        
        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.conv1(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)

        # mep
        feat = [output]
        proj = self.mep_head(feat)     
        # mep

        return output, proj
    
class DEEPLABV3_ASP_MEP_BE(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(DEEPLABV3_ASP_MEP_BE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        
        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.conv1(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        # mep
        feat = [x]
        proj = self.mep_head(feat)     
        # mep

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output, proj

class OCR_ASP_0_MEP_BE(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(OCR_ASP_0_MEP_BE, self).__init__()
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Context
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 1, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, features)


    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        context = self.context[2](feat1, proxy_feats)
        out = context
        output = self.conv_bn_dropout(out)

        # mep
        feat = [x]
        proj = self.mep_head(feat)     
        # mep

        return output, proj

class OCR_ASP_0_MEP_AF(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(OCR_ASP_0_MEP_AF, self).__init__()
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Context
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 1, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)


    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        context = self.context[2](feat1, proxy_feats)
        out = context
        output = self.conv_bn_dropout(out)

        # mep
        feat = [output]
        proj = self.mep_head(feat)     
        # mep

        return output, proj

class OCR_ASP_0_MEP_IN(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(OCR_ASP_0_MEP_IN, self).__init__()
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Context
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 1, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)


    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        context = self.context[2](feat1, proxy_feats)

        # mep
        feat = [feat1]
        proj = self.mep_head(feat)     
        # mep

        out = context

        output = self.conv_bn_dropout(out)
        return output, proj
    
class OCR_ASP_0(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(OCR_ASP_0, self).__init__()
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Context
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 1, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        context = self.context[2](feat1, proxy_feats)

        out = context

        output = self.conv_bn_dropout(out)
        return output

class FCN_ASP_0(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_0, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 1, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        out = feat1

        output = self.conv_bn_dropout(out)
        return output

class FCN_ASP_0_Mep(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_0_Mep, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 1, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        # mep
        feat = [feat1]
        proj = self.mep_head(feat)     
        # mep

        out = feat1

        output = self.conv_bn_dropout(out)
        return output, proj


class FCN_ASP_1_Mep(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_1_Mep, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        # self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        # self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        # self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 2, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        # feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        # feat4 = self.conv4(x)
        # feat5 = self.conv5(x)

        # mep
        feat = [feat1, feat3]
        proj = self.mep_head(feat)     
        # mep

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat3), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat3)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output, proj

class FCN_ASP_2_Mep(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_2_Mep, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        # self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        # self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 3, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        # feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        # feat5 = self.conv5(x)

        # mep
        feat = [feat1, feat3, feat4]
        proj = self.mep_head(feat)     
        # mep

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat3, feat4), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat3, feat4)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output, proj

class FCN_ASP_4_Mep(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_4_Mep, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=48, dilation=48, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        # mep
        feat = [feat2,feat3,feat4,feat5]
        proj = self.mep_head(feat)     
        # mep

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output, proj

class FCN_ASP_3_Mep(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(18, 36, 54), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_3_Mep, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        # self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 4, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        # feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        # mep
        feat = [feat1,feat3,feat4,feat5]
        proj = self.mep_head(feat)     
        # mep

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output, proj


class FCN512_ASP_3_Mep(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(18, 36, 54), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN512_ASP_3_Mep, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features*2, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features*2, bn_type=bn_type),
                                    )
        # self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

        # mep
        from lib.models.nets.contrast_mep import Mep_Module
        self.mep_head = Mep_Module(configer, hidden_features)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        # feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        # mep
        feat = [feat1,feat3,feat4,feat5]
        proj = self.mep_head(feat)     
        # mep

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output, proj

class FCN_ASP_3(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP_3, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        # self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
        #                            ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 4, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        # feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)


        if isinstance(x, Variable):
            # out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
            out = torch.cat((feat1, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            # out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
            out = self._cat_each(feat1, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
    
class DEEPLABV3_ASP(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(DEEPLABV3_ASP, self).__init__()
        # self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                             nn.Conv2d(features, hidden_features, kernel_size=1, bias=False),
        #                             ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv1 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        # feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear',
        #                       align_corners=False)
        feat1 = self.conv1(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)


        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
    
class FCN_ASP(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(FCN_ASP, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.fcn(x)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)


        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output

class OCR_ASP(nn.Module):
    def __init__(self, configer, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(OCR_ASP, self).__init__()
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Context
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)


    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        feat1 = self.context[2](feat1, proxy_feats)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
    
