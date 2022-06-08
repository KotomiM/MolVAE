import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor, requires_grad=requires_grad).cuda()

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def avg_pool(all_vecs, scope, dim):
    size = create_var(torch.Tensor([le for _,le in scope]))
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)

def get_accuracy_bin(scores, labels):
    preds = torch.ge(scores, 0).long()
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()

def get_accuracy(scores, labels):
    _,preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()

def get_accuracy_sym(scores, labels):
    max_scores,max_idx = torch.max(scores, dim=-1)
    lab_scores = scores[torch.arange(len(scores)), labels]
    acc = torch.eq(lab_scores, max_scores).float()
    return torch.sum(acc) / labels.nelement()

def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)

def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)

def zip_tensors(tup_list):
    res = []
    tup_list = zip(*tup_list)
    for a in tup_list:
        if type(a[0]) is int: 
            res.append( torch.LongTensor(a).cuda() )
        else:
            res.append( torch.stack(a, dim=0) )
    return res

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf

def hier_topk(cls_scores, icls_scores, vocab, topk):
    batch_size = len(cls_scores)
    cls_scores = F.log_softmax(cls_scores, dim=-1)
    cls_scores_topk, cls_topk = cls_scores.topk(topk, dim=-1)
    final_topk = []
    for i in range(topk):
        clab = cls_topk[:, i]
        mask = vocab.get_mask(clab)
        masked_icls_scores = F.log_softmax(icls_scores + mask, dim=-1)
        icls_scores_topk, icls_topk = masked_icls_scores.topk(topk, dim=-1)
        topk_scores = cls_scores_topk[:, i].unsqueeze(-1) + icls_scores_topk
        final_topk.append( (topk_scores, clab.unsqueeze(-1).expand(-1, topk), icls_topk) )

    topk_scores, cls_topk, icls_topk = zip(*final_topk)
    topk_scores = torch.cat(topk_scores, dim=-1)
    cls_topk = torch.cat(cls_topk, dim=-1)
    icls_topk = torch.cat(icls_topk, dim=-1)

    topk_scores, topk_index = topk_scores.topk(topk, dim=-1)
    batch_index = cls_topk.new_tensor([[i] * topk for i in range(batch_size)])
    cls_topk = cls_topk[batch_index, topk_index]
    icls_topk = icls_topk[batch_index, topk_index]
    return topk_scores, cls_topk.tolist(), icls_topk.tolist()

#3D padded tensor to 2D matrix, with padded zeros removed
def flatten_tensor(tensor, scope):
    assert tensor.size(0) == len(scope)
    tlist = []
    for i,tup in enumerate(scope):
        le = tup[1]
        tlist.append( tensor[i, 0:le] )
    return torch.cat(tlist, dim=0)

#2D matrix to 3D padded tensor
def inflate_tensor(tensor, scope): 
    max_len = max([le for _,le in scope])
    batch_vecs = []
    for st,le in scope:
        cur_vecs = tensor[st : st + le]
        cur_vecs = F.pad( cur_vecs, (0,0,0,max_len-le) )
        batch_vecs.append( cur_vecs )

    return torch.stack(batch_vecs, dim=0)

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x,sum_h], dim=1)
    z = F.sigmoid(W_z(z_input))

    r_1 = W_r(x).view(-1,1,hidden_size)
    r_2 = U_r(h_nei)
    r = F.sigmoid(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x,sum_gated_h], dim=1)
    pre_h = F.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h