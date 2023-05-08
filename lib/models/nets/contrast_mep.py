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
        self.memory_size = self.configer.get("contrast", "memory_size")
        self.num_classes = self.configer.get("data", "num_classes")


        for layer in self.projector:
            if layer == "layer_4":
                self.projector_layer4 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
                
                if self.memory_size:
                    self.register_buffer(layer+"_queue", nn.functional.normalize(torch.randn(1, self.memory_size, self.proj_dim), p=2, dim=2))
                    self.register_buffer(layer+"_queue_label", torch.randint(0, self.num_classes, (1, self.memory_size),dtype=torch.long))
                    self.register_buffer(layer+"_queue_ptr", torch.zeros(1, dtype=torch.long))
            elif layer == "layer_3":
                self.projector_layer3 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
                
                if self.memory_size:
                    self.register_buffer(layer+"_queue", nn.functional.normalize(torch.randn(1, self.memory_size, self.proj_dim), p=2, dim=2))
                    self.register_buffer(layer+"_queue_label", torch.randint(0, self.num_classes, (1, self.memory_size),dtype=torch.long))
                    self.register_buffer(layer+"_queue_ptr", torch.zeros(1, dtype=torch.long))
            elif layer == "layer_2":
                self.projector_layer2 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
                
                if self.memory_size:
                    self.register_buffer(layer+"_queue", nn.functional.normalize(torch.randn(1, self.memory_size, self.proj_dim), p=2, dim=2))
                    self.register_buffer(layer+"_queue_label", torch.randint(0, self.num_classes, (1, self.memory_size),dtype=torch.long))
                    self.register_buffer(layer+"_queue_ptr", torch.zeros(1, dtype=torch.long))
            elif layer == "layer_1":
                self.projector_layer1 = nn.Sequential(
                    nn.Conv2d(inout_dim, inout_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    ModuleHelper.BNReLU(inout_dim, bn_type=self.configer.get("network", "bn_type")),
                    nn.Conv2d(inout_dim, self.proj_dim, kernel_size=1, stride=1, padding=0, bias=True),)
                
                if self.memory_size:
                    self.register_buffer(layer+"_queue", nn.functional.normalize(torch.randn(1, self.memory_size, self.proj_dim), p=2, dim=2))
                    self.register_buffer(layer+"_queue_label", torch.randint(0, self.num_classes, (1, self.memory_size),dtype=torch.long))
                    self.register_buffer(layer+"_queue_ptr", torch.zeros(1, dtype=torch.long))
            
    

    def forward(self, feat):
        output = dict()
        layer = dict()

        for lay in self.projector:
            if lay == 'layer_4':
                proj_layer4 = F.normalize(self.projector_layer4(feat[3]), dim=1)

                queue_layer4 = None
                if self.memory_size:
                    queue_layer4 = [self.layer_4_queue, self.layer_4_queue_label, self.layer_4_queue_ptr]

                layer['layer_4'] = [proj_layer4, queue_layer4]
            elif lay == 'layer_3':
                proj_layer3 = F.normalize(self.projector_layer3(feat[2]), dim=1)

                queue_layer3 = None
                if self.memory_size:
                    queue_layer3 = [self.layer_3_queue, self.layer_3_queue_label, self.layer_3_queue_ptr]

                layer['layer_3'] = [proj_layer3, queue_layer3]
            elif lay == 'layer_2':
                proj_layer2 = F.normalize(self.projector_layer2(feat[1]), dim=1)

                queue_layer2 = None
                if self.memory_size:
                    queue_layer2 = [self.layer_2_queue, self.layer_2_queue_label, self.layer_2_queue_ptr]
                    
                layer['layer_2'] = [proj_layer2, queue_layer2]
            elif lay == 'layer_1':
                proj_layer1 = F.normalize(self.projector_layer1(feat[0]), dim=1)

                queue_layer1 = None
                if self.memory_size:
                    queue_layer1 = [self.layer_1_queue, self.layer_1_queue_label, self.layer_1_queue_ptr]
                    
                layer['layer_1'] = [proj_layer1, queue_layer1]

        output["proj"] = layer

        return output