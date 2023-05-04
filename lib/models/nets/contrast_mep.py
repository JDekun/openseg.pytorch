import os
import pdb
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class Mep_Module(nn.Module):
    def __init__(self, 
                 configer,
                 inout_dim,
                 ):
        super(Mep_Module, self).__init__()

        self.configer = configer
        self.proj_dim = self.configer.get("contrast", "proj_dim")
        self.projector = self.configer.get("contrast", "projector")


        for layer in self.projector:
            if layer == "layer_4":
                self.projector_layer4 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
            elif layer == "layer_3":
                self.projector_layer3 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
            elif layer == "layer_2":
                self.projector_layer2 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
    

    def forward(self, feat3, feat4, feat5):
        output = dict()
        layer = dict()

        for lay in self.projector:
            if lay == 'layer_4':
                proj_layer4 = F.normalize(self.projector_layer4(feat5), dim=1)
                layer['layer_4'] = proj_layer4
            elif lay == 'layer_3':
                proj_layer3 = F.normalize(self.projector_layer3(feat4), dim=1)
                layer['layer_3'] = proj_layer3
            elif lay == 'layer_2':
                proj_layer2 = F.normalize(self.projector_layer2(feat3), dim=1)
                layer['layer_2'] = proj_layer2

        output["proj"] = layer

        return output