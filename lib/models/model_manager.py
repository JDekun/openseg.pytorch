##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FCN baseline
from lib.models.nets.fcnet import FcnNet

# OCR
from lib.models.nets.ocrnet import SpatialOCRNet, ASPOCRNet
from lib.models.nets.ideal_ocrnet import (
    IdealSpatialOCRNet,
    IdealSpatialOCRNetB,
    IdealSpatialOCRNetC,
    IdealGatherOCRNet,
    IdealDistributeOCRNet,
)

# HRNet
from lib.models.nets.hrnet import HRNet_W48
from lib.models.nets.hrnet import HRNet_W48_OCR, HRNet_W48_ASPOCR, HRNet_W48_OCR_B

# OCNet
from lib.models.nets.ocnet import BaseOCNet, AspOCNet

# ISA Net
from lib.models.nets.isanet import ISANet

# CE2P
from lib.models.nets.ce2pnet import CE2P_OCRNet, CE2P_IdealOCRNet, CE2P_ASPOCR

# SegFix
from lib.models.nets.segfix import SegFix_HRNet

# dcnet
from lib.models.nets.fcnet_dc import FcnNetDC
from lib.models.nets.ocrnet_dc import SpatialOCRNetDC
from lib.models.nets.ocrnet_dc import ASPOCRNetMep
from lib.models.nets.hrnet_dc import HRNet_W48_DC
from lib.models.nets.hrnet_dc import HRNet_W48_OCR_DC
from lib.models.nets.hrnet_dc import HRNet_W48_ASPOCR_Mep
# contrast
from lib.models.nets.contrast_hrnet import HRNet_W48_ASPOCR_MLM

# contrast_asp
from lib.models.nets.contrast_resnet import RES_OCR_ASP
from lib.models.nets.contrast_resnet import RES_FCN_ASP
from lib.models.nets.contrast_resnet import RES_FCN_ASP_3
from lib.models.nets.contrast_resnet import RES_FCN_ASP_3_MEP
from lib.models.nets.contrast_resnet import RES_FCN_ASP_2_MEP
from lib.models.nets.contrast_resnet import RES_FCN_ASP_1_MEP
from lib.models.nets.contrast_resnet import RES_FCN_ASP_0_MEP
from lib.models.nets.contrast_resnet import RES_OCR_ASP_0_MEP_IN
from lib.models.nets.contrast_resnet import RES_OCR_ASP_0_MEP_AF
from lib.models.nets.contrast_resnet import RES_OCR_ASP_0_MEP_BE


from lib.utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    # SegFix
    "segfix_hrnet": SegFix_HRNet,
    # OCNet series
    "base_ocnet": BaseOCNet,
    "asp_ocnet": AspOCNet,
    # ISA Net
    "isanet": ISANet,
    # OCR series
    "spatial_ocrnet": SpatialOCRNet,
    "spatial_asp_ocrnet": ASPOCRNet,
    # OCR series with ground-truth
    "ideal_spatial_ocrnet": IdealSpatialOCRNet,
    "ideal_spatial_ocrnet_b": IdealSpatialOCRNetB,
    "ideal_spatial_ocrnet_c": IdealSpatialOCRNetC,
    "ideal_gather_ocrnet": IdealGatherOCRNet,
    "ideal_distribute_ocrnet": IdealDistributeOCRNet,
    # HRNet series
    "hrnet_w48": HRNet_W48,
    "hrnet_w48_ocr": HRNet_W48_OCR,
    "hrnet_w48_ocr_b": HRNet_W48_OCR_B,
    "hrnet_w48_asp_ocr": HRNet_W48_ASPOCR,
    # CE2P series
    "ce2p_asp_ocrnet": CE2P_ASPOCR,
    "ce2p_ocrnet": CE2P_OCRNet,
    "ce2p_ideal_ocrnet": CE2P_IdealOCRNet,
    # baseline series
    "fcnet": FcnNet,
    # dcnet
    "spatial_ocrnet_dc": SpatialOCRNetDC,
    "fcnet_dc": FcnNetDC,
    "hrnet_w48_dc": HRNet_W48_DC,
    "hrnet_w48_ocr_dc": HRNet_W48_OCR_DC,
    "spatial_asp_ocrnet_mep": ASPOCRNetMep,
    "hrnet_w48_asp_ocr_mep": HRNet_W48_ASPOCR_Mep,
    # mep
    "hrnet_w48_asp_ocr_mlm": HRNet_W48_ASPOCR_MLM,
    # contrast_asp
    "resnet_ocr_asp": RES_OCR_ASP,
    "resnet_fcn_asp": RES_FCN_ASP,
    "resnet_fcn_asp3": RES_FCN_ASP_3,
    "resnet_fcn_asp3_mep": RES_FCN_ASP_3_MEP,
    "resnet_fcn_asp2_mep": RES_FCN_ASP_2_MEP,
    "resnet_fcn_asp1_mep": RES_FCN_ASP_1_MEP,
    "resnet_fcn_asp0_mep": RES_FCN_ASP_0_MEP,
    "resnet_ocr_asp0_mep_in": RES_OCR_ASP_0_MEP_IN,
    # "resnet_ocr_asp0_mep_af": RES_OCR_ASP_0_MEP_AF,
    # "resnet_ocr_asp0_mep_be": RES_OCR_ASP_0_MEP_BE,
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get("network", "model_name")

        if model_name not in SEG_MODEL_DICT:
            Log.error("Model: {} not valid!".format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
