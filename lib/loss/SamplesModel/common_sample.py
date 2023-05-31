import torch

def label_ave(this_y_hat, this_y, cls_id):
    indices = (this_y_hat == cls_id).nonzero()
    
    return indices

def pred_ave(this_y_hat, this_y, cls_id):
    indices = (this_y != cls_id).nonzero()

    return indices

def only_esay(this_y_hat, this_y, cls_id):
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

    return easy_indices

def only_esay_one(this_y_hat, this_y, cls_id):
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    
    num_easy = easy_indices.shape[0]
    perm = torch.randperm(num_easy)
    easy_indices = easy_indices[perm[:1]]

    return easy_indices