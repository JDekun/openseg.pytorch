import os
import pdb
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper

class Contrast_Module(nn.Module):
    def __init__(self, 
                 configer,
                 ):
        super(Contrast_Module, self).__init__()

        self.configer = configer
        self.proj_dim = self.configer.get("contrast", "proj_dim")
        self.projector = self.configer.get("contrast", "projector")

        # >>> project contrast
        self.projector_decode = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
            nn.Conv2d(512, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
            ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
        if "resnet" in self.configer.get("network", "backbone"):
            for layer in self.projector:
                if layer == "layer_4":
                    self.projector_layer4 = nn.Sequential(
                        nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(512, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                elif layer == "layer_3":
                    self.projector_layer3 = nn.Sequential(
                        nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(512, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                elif layer == "layer_2":
                    self.projector_layer2 = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(512, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(512, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                elif layer == "layer_1":
                    self.projector_layer1 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(256, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(256, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
        elif "hrnet" in self.configer.get("network", "backbone"):
            for layer in self.projector:
                if layer == "layer_4":
                    self.projector_layer4 = nn.Sequential(
                        nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(384, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(384, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                elif layer == "layer_3":
                    self.projector_layer3 = nn.Sequential(
                        nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(192, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(192, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                elif layer == "layer_2":
                    self.projector_layer2 = nn.Sequential(
                        nn.Conv2d(96, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                elif layer == "layer_1":
                    self.projector_layer1 = nn.Sequential(
                        nn.Conv2d(48, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),
                        nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                        ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
                    
        self.de_projector = nn.Sequential(
            nn.Conv2d(self.proj_dim, 512, kernel_size=1, stride=1, padding=0, bias=True),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get("network", "bn_type"))(512),)
        self.relu = nn.ReLU()

    def forward(self, bk, decode):
        output = dict()
        layer = dict()

        temp = self.projector_decode(decode)
        proj_decode = F.normalize(temp, dim=1)
        output["decode"] = proj_decode

        for lay in self.projector:
            if lay == 'layer_4':
                proj_layer4 = F.normalize(self.projector_layer4(bk[-1]), dim=1)
                layer['layer_4'] = proj_layer4
            elif lay == 'layer_3':
                proj_layer3 = F.normalize(self.projector_layer3(bk[-2]), dim=1)
                layer['layer_3'] = proj_layer3
            elif lay == 'layer_2':
                proj_layer2 = F.normalize(self.projector_layer2(bk[-3]), dim=1)
                layer['layer_2'] = proj_layer2
            elif lay == 'layer_1':
                proj_layer1 = F.normalize(self.projector_layer1(bk[-4]), dim=1)
                layer['layer_1'] = proj_layer1

        output["proj"] = layer

        contrast = self.de_projector(temp)
        feats = self.relu(decode + contrast)

        return output, feats
    

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
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
            elif layer == "layer_3":
                self.projector_layer3 = nn.Sequential(
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
            elif layer == "layer_2":
                self.projector_layer2 = nn.Sequential(
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
            elif layer == "layer_1":
                self.projector_layer1 = nn.Sequential(
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(self.proj_dim, bn_type=self.configer.get("network", "bn_type")),)
        for layer in self.projector:
            if layer == "layer_4":
                self.de_projector_layer4 = nn.Sequential(
                    nn.Conv2d(self.proj_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),)
            elif layer == "layer_3":
                self.de_projector_layer3 = nn.Sequential(
                    nn.Conv2d(self.proj_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),)
            elif layer == "layer_2":
                self.de_projector_layer2 = nn.Sequential(
                    nn.Conv2d(self.proj_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),)
            elif layer == "layer_1":
                self.de_projector_layer1 = nn.Sequential(
                    nn.Conv2d(self.proj_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),)
                

    def forward(self, feat2, feat3, feat4, feat5):
        output = dict()
        layer = dict()

        for lay in self.projector:
            if lay == 'layer_4':
                proj_layer4 = F.normalize(self.projector_layer4(feat5), dim=1)
                layer['layer_4'] = proj_layer4
                feat5 = self.de_projector_layer4(proj_layer4)
            elif lay == 'layer_3':
                proj_layer3 = F.normalize(self.projector_layer3(feat4), dim=1)
                layer['layer_3'] = proj_layer3
                feat4 = self.de_projector_layer3(proj_layer3)
            elif lay == 'layer_2':
                proj_layer2 = F.normalize(self.projector_layer2(feat3), dim=1)
                layer['layer_2'] = proj_layer2
                feat3 = self.de_projector_layer2(proj_layer2)
            elif lay == 'layer_1':
                proj_layer1 = F.normalize(self.projector_layer1(feat2), dim=1)
                layer['layer_1'] = proj_layer1
                feat2 = self.de_projector_layer1(proj_layer1)

        output["proj"] = layer

        return feat2, feat3, feat4, feat5, output