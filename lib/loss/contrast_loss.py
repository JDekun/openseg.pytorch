import torch
import torch.nn as nn
from .SamplesModel import Sampling

def sample_negative(Q, Q_label):
    class_num, cache_size, feat_size = Q.shape

    X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
    y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
    sample_ptr = 0
    for ii in range(class_num):
        # if ii == 0: continue
        this_q = Q[ii, :cache_size, :]
        # this_q_label = Q_label[ii, :cache_size]

        X_[sample_ptr:sample_ptr + cache_size, :] = this_q
        y_[sample_ptr:sample_ptr + cache_size, :] = torch.transpose(Q_label, 0, 1)
        sample_ptr += cache_size

    return X_, y_ 


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

def Contrastive(feats_x, feats_y, labels_, queue=None, queue_label=None, type: str = 'intra', temperature: float = 0.1, base_temperature: float = 0.07):
    anchor_num, n_view = feats_x.shape[0], feats_x.shape[1]

    feature_x = torch.cat(torch.unbind(feats_x, dim=1), dim=0)
    feature_y = torch.cat(torch.unbind(feats_y, dim=1), dim=0)

    if type == "inter":
        anchor_feature = feature_x
        contrast_feature= feature_y
        anchor_count = n_view
        contrast_count = n_view
    elif type == "intra":
        anchor_feature = feature_x
        contrast_feature= feature_x
        anchor_count = n_view
        contrast_count = n_view
    elif type == "double":
        anchor_feature = torch.cat([feature_x, feature_y], dim=0)
        contrast_feature = anchor_feature
        anchor_count = n_view * 2
        contrast_count = n_view * 2
         
    # 基础mask
    labels_ = labels_.contiguous().view(-1, 1)
    labels_T = labels_
    mask = torch.eq(labels_, torch.transpose(labels_T, 0, 1)).float().cuda()
    mask = mask.repeat(anchor_count, contrast_count)
    
    if queue is not None:
        queue_feature, queue_label = sample_negative(queue, queue_label) # 并行队列变形成串行

        # 增加queue特征
        contrast_feature = queue_feature

        # 增加queue mask
        queue_label = queue_label.contiguous().view(-1, 1)
        mask_queue = torch.eq(labels_, torch.transpose(queue_label, 0, 1)).float().cuda()
        mask_queue = mask_queue.repeat(anchor_count, 1)

        # 更新mask
        mask = mask_queue

    # 计算对比logits
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # logits = anchor_dot_contrast

    if (type == "inter") or (queue is not None):
        ops_mask = mask
    else:
        # mask对角线logits(自身对比部分)
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                    torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                    0)
        # 正样本mask
        ops_mask = mask * logits_mask


    # 负样本mask
    neg_mask = 1 - mask

    # 负样本对比总和
    exp_logits = torch.exp(logits)
    neg_logits = exp_logits * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    # 防止出现都正样本个数为0的情况
    ops_mask_num = ops_mask.sum(1)
    for i in range(len(ops_mask_num)):
        if ops_mask_num[i] == 0:
            ops_mask_num[i] = 1 

    # 计算对比损失
    log_prob = logits - torch.log(exp_logits + neg_logits)
    mean_log_prob_pos = (ops_mask * log_prob).sum(1) / ops_mask_num
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def CONTRAST_Loss(cls_score,
                decode,
                layer,
                queue_origin,
                labels,
                memory_size = 0,
                sample = 'weight_ade_8',
                contrast_type = None):
    
    if decode == None:
        decode = layer
    feats = decode
    feats_y = layer

    h, w = feats.shape[2], feats.shape[3]
    pred = torch.nn.functional.interpolate(input=cls_score, size=(h, w), mode='bilinear', align_corners=False)
    _, predict = torch.max(pred, 1)

    labels = labels.unsqueeze(1).float().clone()
    labels = torch.nn.functional.interpolate(labels, (h, w), mode='nearest')
    labels = labels.squeeze(1).long()
    assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)


    batch_size = feats.shape[0]

    labels = labels.contiguous().view(batch_size, -1)
    predict = predict.contiguous().view(batch_size, -1)

    feats = feats.permute(0, 2, 3, 1)
    feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
    feats_y = feats_y.permute(0, 2, 3, 1)
    feats_y = feats_y.contiguous().view(feats_y.shape[0], -1, feats_y.shape[-1])


    feats_, feats_y_, labels_, feats_que_, feats_y_que_, labels_queue_ = Sampling(sample, feats, feats_y, labels, predict)

    if feats_ != None:
        if memory_size:
            for i in range(len(queue_origin)):
                if i == 0:
                    queue = queue_origin[i][0]
                    queue_label = queue_origin[i][1]
                else:
                    queue = torch.cat([queue, queue_origin[i][0]], dim=1)
                    queue_label = torch.cat([queue_label, queue_origin[i][1]], dim=1)

            loss = Contrastive(feats_, feats_y_, labels_, queue,  queue_label, contrast_type)

        else:
            queue=None
            queue_label=None

            loss = Contrastive(feats_, feats_y_, labels_, queue,  queue_label, contrast_type)
        
        # if memory_size:
        #     dequeue_and_enqueue_self_seri(feats_que_, feats_y_que_, labels_queue_,
        #                                     encode_queue=queue_origin[0],
        #                                     code_queue_label=queue_origin[1],
        #                                     encode_queue_ptr=queue_origin[2])
    else:
        loss = 0

    return loss, feats_que_, feats_y_que_, labels_queue_