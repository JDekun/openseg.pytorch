##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.tools.logger import Logger as Log
from lib.loss.rmi_loss import RMILoss

from .contrast_loss import CONTRAST_Loss


class WeightedFSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super().__init__()
        self.configer = configer
        self.thresh = self.configer.get("loss", "params")["ohem_thresh"]
        self.reduction = "elementwise_mean"
        if self.configer.exists(
            "loss", "params"
        ) and "ce_reduction" in self.configer.get("loss", "params"):
            self.reduction = self.configer.get("loss", "params")["ce_reduction"]

    def forward(
        self, predict, target, min_kept=1, weight=None, ignore_index=-1, **kwargs
    ):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = (
            target.contiguous().view(
                -1,
            )
            != ignore_index
        )
        sort_prob, sort_indices = (
            prob.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )
        min_threshold = sort_prob[min(min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matrix = (
            F.cross_entropy(
                predict,
                target,
                weight=weight,
                ignore_index=ignore_index,
                reduction="none",
            )
            .contiguous()
            .view(
                -1,
            )
        )
        sort_loss_matrix = loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]
        if self.reduction == "sum":
            return select_loss_matrix.sum()
        elif self.reduction == "elementwise_mean":
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError("Reduction Error!")


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists("loss", "params") and "ce_weight" in self.configer.get(
            "loss", "params"
        ):
            weight = self.configer.get("loss", "params")["ce_weight"]
            weight = torch.FloatTensor(weight).cuda()

        reduction = "elementwise_mean"
        if self.configer.exists(
            "loss", "params"
        ) and "ce_reduction" in self.configer.get("loss", "params"):
            reduction = self.configer.get("loss", "params")["ce_reduction"]

        ignore_index = -1
        if self.configer.exists(
            "loss", "params"
        ) and "ce_ignore_index" in self.configer.get("loss", "params"):
            ignore_index = self.configer.get("loss", "params")["ce_ignore_index"]

        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(
                        targets[i], (inputs[i].size(2), inputs[i].size(3))
                    )
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(
                        targets[0], (inputs[i].size(2), inputs[i].size(3))
                    )
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode="nearest")
        return targets.squeeze(1).long()


class FSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super(FSOhemCELoss, self).__init__()
        self.configer = configer
        self.thresh = self.configer.get("loss", "params")["ohem_thresh"]
        self.min_kept = max(1, self.configer.get("loss", "params")["ohem_minkeep"])
        weight = None
        if self.configer.exists("loss", "params") and "ce_weight" in self.configer.get(
            "loss", "params"
        ):
            weight = self.configer.get("loss", "params")["ce_weight"]
            weight = torch.FloatTensor(weight).cuda()

        self.reduction = "elementwise_mean"
        if self.configer.exists(
            "loss", "params"
        ) and "ce_reduction" in self.configer.get("loss", "params"):
            self.reduction = self.configer.get("loss", "params")["ce_reduction"]

        ignore_index = -1
        if self.configer.exists(
            "loss", "params"
        ) and "ce_ignore_index" in self.configer.get("loss", "params"):
            ignore_index = self.configer.get("loss", "params")["ce_ignore_index"]

        self.ignore_label = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction="none"
        )

    def forward(self, predict, target, **kwargs):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = (
            target.contiguous().view(
                -1,
            )
            != self.ignore_label
        )
        mask[0] = 1  # Avoid `mask` being empty
        sort_prob, sort_indices = (
            prob.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matirx = (
            self.ce_loss(predict, target)
            .contiguous()
            .view(
                -1,
            )
        )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == "sum":
            return select_loss_matrix.sum()
        elif self.reduction == "elementwise_mean":
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError("Reduction Error!")


class FSAuxOhemCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxOhemCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        if self.configer.get("loss", "loss_type") == "fs_auxohemce_loss":
            self.ohem_ce_loss = FSOhemCELoss(self.configer)
        else:
            assert self.configer.get("loss", "loss_type") == "fs_auxslowohemce_loss"
            self.ohem_ce_loss = FSSlowOhemCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get("network", "loss_weights")["seg_loss"] * seg_loss
        loss = (
            loss + self.configer.get("network", "loss_weights")["aux_loss"] * aux_loss
        )
        return loss


class FSAuxCELossDC(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELossDC, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out, proj = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        
        # contrast
        cls_score = seg_out
        conX=None
        if "decode" in proj:
            conX = proj['decode']
        los_con = 0
        memory_size = self.configer.get("contrast", "memory_size")
        
        if memory_size:
            i = 0
            mbank = []
            for name, conY in proj['proj'].items():
                mbank[i] = conY[1]
                i = i + 1

        feats_que = []
        feats_y_que = []
        labels_queue = []
        n = 0
        for name, conY in proj['proj'].items():
            index = int(name.split("_")[-1]) - 1
            weight = self.configer.get("contrast", "loss_weights")[index]
            
            # mbank = conY[1]
            con, feats_que_, feats_y_que_, labels_queue_ = CONTRAST_Loss(
                cls_score,
                conX,
                conY[0],
                mbank,
                targets,
                memory_size,
                sample = 'weight_ade_8',
                contrast_type = self.configer.get("contrast", "contrast_type"))
            los_con = los_con + weight * con
            feats_que[n] = feats_que_
            feats_y_que[n] = feats_y_que_
            labels_queue[n] = labels_queue_
            n = n + 1
        # contrast

        if memory_size:
            for i in range(len(mbank)):
                dequeue_and_enqueue_self_seri(feats_que[i], feats_y_que[i], labels_queue[i],
                                                encode_queue=mbank[i][0],
                                                code_queue_label=mbank[i][1],
                                                encode_queue_ptr=mbank[i][2])

        loss = self.configer.get("network", "loss_weights")["seg_loss"] * seg_loss
        loss = loss + los_con
        loss = (
            loss + self.configer.get("network", "loss_weights")["aux_loss"] * aux_loss
        )
        return loss


def dequeue_and_enqueue_self_seri(keys, key_y, labels,
                                encode_queue, code_queue_label, encode_queue_ptr
                                ):
    memory_size = encode_queue.shape[1]

    iter =  len(labels)
    for i in range(iter):
        lb = 0
        lbe = int(labels[i])
        feat = keys[i]
        feat_y = key_y[i]
        K = feat.shape[0]

        ptr = int(encode_queue_ptr[lb])

        if ptr + K > memory_size:
            total = ptr + K
            start = total - memory_size
            end = K - start

            encode_queue[lb, ptr:memory_size, :] = feat[0:end]
            encode_queue[lb, 0:start, :] = feat[end:]
            encode_queue_ptr[lb] = start

            code_queue_label[lb, ptr:memory_size] = lbe
            code_queue_label[lb, 0:start] = lbe

        else:
            encode_queue[lb, ptr:ptr + K, :] = feat
            encode_queue_ptr[lb] = (encode_queue_ptr[lb] + K) % memory_size

            code_queue_label[lb, ptr:ptr + K] = lbe

class FSCELossDC(nn.Module):
    def __init__(self, configer=None):
        super(FSCELossDC, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        seg_out, proj = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        
        cls_score = seg_out
        decode = proj['decode']
        los_con = 0
        for name, layer in proj['proj'].items():
            index = int(name.split("_")[-1]) - 1
            weight = self.configer.get("contrast", "loss_weights")[index]
            con = CONTRAST_Loss(
                cls_score,
                decode,
                layer,
                targets,
                memory_size = 0,
                sample = 'weight_ade_8')
            los_con = los_con + weight * con

        loss = self.configer.get("network", "loss_weights")["seg_loss"] * seg_loss
        loss = loss + los_con
        loss = (
            loss
        )
        return loss


class FSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxRMILoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = self.configer.get("network", "loss_weights")["seg_loss"] * seg_loss
        loss = (
            loss + self.configer.get("network", "loss_weights")["aux_loss"] * aux_loss
        )
        return loss


class SegFixLoss(nn.Module):
    """
    We predict a binary mask to categorize the boundary pixels as class 1 and otherwise as class 0
    Based on the pixels predicted as 1 within the binary mask, we further predict the direction for these
    pixels.
    """

    def __init__(self, configer=None):
        super().__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def calc_weights(self, label_map, num_classes):

        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()

    def forward(self, inputs, targets, **kwargs):

        from lib.utils.helpers.offset_helper import DTOffsetHelper

        pred_mask, pred_direction = inputs

        seg_label_map, distance_map, angle_map = targets[0], targets[1], targets[2]
        gt_mask = DTOffsetHelper.distance_to_mask_label(
            distance_map, seg_label_map, return_tensor=True
        )

        gt_size = gt_mask.shape[1:]
        mask_weights = self.calc_weights(gt_mask, 2)

        pred_direction = F.interpolate(
            pred_direction, size=gt_size, mode="bilinear", align_corners=True
        )
        pred_mask = F.interpolate(
            pred_mask, size=gt_size, mode="bilinear", align_corners=True
        )
        mask_loss = F.cross_entropy(
            pred_mask, gt_mask, weight=mask_weights, ignore_index=-1
        )

        mask_threshold = float(os.environ.get("mask_threshold", 0.5))
        binary_pred_mask = torch.softmax(pred_mask, dim=1)[:, 1, :, :] > mask_threshold

        gt_direction = DTOffsetHelper.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label_map,
            extra_ignore_mask=(binary_pred_mask == 0),
            return_tensor=True,
        )

        direction_loss_mask = gt_direction != -1
        direction_weights = self.calc_weights(
            gt_direction[direction_loss_mask], pred_direction.size(1)
        )
        direction_loss = F.cross_entropy(
            pred_direction, gt_direction, weight=direction_weights, ignore_index=-1
        )

        if (
            self.training
            and self.configer.get("iters") % self.configer.get("solver", "display_iter")
            == 0
            and torch.cuda.current_device() == 0
        ):
            Log.info(
                "mask loss: {} direction loss: {}.".format(mask_loss, direction_loss)
            )

        mask_weight = float(os.environ.get("mask_weight", 1))
        direction_weight = float(os.environ.get("direction_weight", 1))

        return mask_weight * mask_loss + direction_weight * direction_loss
