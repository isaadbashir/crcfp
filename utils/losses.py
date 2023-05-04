import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ramps


class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


class EntropyMinimization(nn.Module):
    def __init__(self, reduction='mean'):
        super(EntropyMinimization, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):
        P = torch.softmax(inputs, dim=1)
        logP = torch.log_softmax(inputs, dim=1)
        PlogP = P * logP
        loss_ent = -1.0 * PlogP.sum(dim=1)
        if self.reduction == 'mean':
            loss_ent = torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = torch.sum(loss_ent)
        else:
            pass

        return loss_ent


class AverageEntropyMinimization(nn.Module):
    def __init__(self, reduction='mean'):
        super(AverageEntropyMinimization, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):

        # compute the softmax
        probs = torch.softmax(inputs, dim=1)

        # average across the batch
        mean_prob = torch.mean(probs, dim=0)

        # find the PlogP
        PlogP = torch.log(mean_prob ** mean_prob)

        # sum across the channels again
        loss_ent = -1.0 * PlogP.mean(dim=1)

        if self.reduction == 'mean':
            loss_ent = torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = torch.sum(loss_ent)
        else:
            pass

        return loss_ent


class AverageEntropyMaximization(nn.Module):
    def __init__(self, reduction='mean'):
        super(AverageEntropyMaximization, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):

        # compute the softmax
        probs = torch.softmax(inputs, dim=1)

        # average across the batch
        mean_prob = torch.mean(probs, dim=0)

        # find the PlogP
        PlogP = torch.log(mean_prob ** mean_prob)

        # sum across the channels again
        loss_ent = -1.0 * PlogP.mean(dim=1)

        if self.reduction == 'mean':
            loss_ent = -torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = -torch.sum(loss_ent)
        else:
            pass

        return loss_ent


class ScaledEntropyMinimization(nn.Module):
    def __init__(self, reduction='mean', gamma=0.1):
        super(ScaledEntropyMinimization, self).__init__()
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, inputs):

        P = torch.softmax(inputs, dim=1)
        P = (1 - 2 * self.gamma) * P + self.gamma
        logP = torch.log_softmax(inputs, dim=1)
        PlogP = P * logP
        loss_ent = -1.0 * PlogP.sum(dim=1)
        if self.reduction == 'mean':
            loss_ent = torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = torch.sum(loss_ent)
        else:
            pass

        return loss_ent


class IWsoftCrossEntropy(nn.Module):
    # class_wise softCrossEntropy for class balance
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss with image-wise weighting factor
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)
        _, argpred = torch.max(inputs, 1)
        weights = []
        batch_size = inputs.size(0)
        for i in range(batch_size):
            hist = torch.histc(argpred[i].cpu().data.float(),
                               bins=self.num_class, min=0,
                               max=self.num_class - 1).float()
            weight = (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.sum((torch.mul(-log_likelihood, target) * weights)[mask]) / (batch_size * self.num_class)
        return loss


class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, prob, label=None):

        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=self.num_class + 1, min=-1,
                               max=self.num_class - 1).float()
            hist = hist[1:]
            weight = (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        weights = weights.unsqueeze(1).expand_as(prob)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * self.num_class)
        return loss


class MaxSquareloss(nn.Module):
    def __init__(self, reduction='sum'):
        super(MaxSquareloss, self).__init__()
        self.reduction = reduction

    def forward(self, prob):
        prob = torch.softmax(prob, dim=1)
        p2 = torch.pow(prob, 2)
        loss_msl = -1.0 * p2.mean(dim=1)

        if self.reduction == 'mean':
            loss_msl = torch.mean(loss_msl) / 2
        elif self.reduction == 'sum':
            loss_msl = torch.sum(loss_msl) / 2
        else:
            pass

        return loss_msl


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


def rand_bbox(size, mu, sigma):

    prob = np.random.normal(mu, sigma)
    W = size[2]
    H = size[3]
    cut_w = int(W * prob)
    cut_h = int(H * prob)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    w = bbx2 - bbx1
    h = bby2 - bby1

    return bbx1, bby1, bbx2, bby2


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """

    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                 reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.5 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]

        # print("select_loss_matrix.size(): {}".format(select_loss_matrix.size()))
        # input()

        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


def softmax_mse_loss(inputs, targets, conf_mask=True, threshold=0.75, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0:
            loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0:
            loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5


def pair_wise_loss(unsup_outputs, size_average=True, nbr_of_pairs=8):
    """
    Pair-wise loss in the sup. mat.
    """
    if isinstance(unsup_outputs, list):
        unsup_outputs = torch.stack(unsup_outputs)

    # Only for a subset of the aux outputs to reduce computation and memory
    unsup_outputs = unsup_outputs[torch.randperm(unsup_outputs.size(0))]
    unsup_outputs = unsup_outputs[:nbr_of_pairs]

    temp = torch.zeros_like(unsup_outputs)  # For grad purposes
    for i, u in enumerate(unsup_outputs):
        temp[i] = F.softmax(u, dim=1)
    mean_prediction = temp.mean(0).unsqueeze(0)  # Mean over the auxiliary outputs
    pw_loss = ((temp - mean_prediction) ** 2).mean(0)  # Variance
    pw_loss = pw_loss.sum(1)  # Sum over classes
    if size_average:
        return pw_loss.mean()
    return pw_loss.sum()


class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss


def softmax_mse_loss(inputs, targets, conf_mask=True, threshold=0.75, use_softmax=False):

    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()  # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0:
            loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')  # take the mean over the batch_size


def update_ema_variables(extractor, extractor_ema, classifier, classifier_ema, projector, projector_ema, alpha, global_step):
    """

    :param extractor:
    :param extractor_ema:
    :param classifier:
    :param classifier_ema:
    :param projector:
    :param projector_ema:
    :param alpha:
    :param global_step:
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(extractor_ema.parameters(), extractor.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(projector_ema.parameters(), projector.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(classifier_ema.parameters(), classifier.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        _targets = targets.clone()
        if mask is not None:
            _targets[mask] = self.ignore_index

        loss = F.cross_entropy(inputs, _targets, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, bdp_threshold, fdp_threshold, temp=0.1, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.bdp_threshold = bdp_threshold
        self.fdp_threshold = fdp_threshold

    def forward(self, anchor, pos_pair, neg_pair, pseudo_label, pseudo_label_all, FC, FC_all):
        pos = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True)

        mask_pixel_filter = (pseudo_label.unsqueeze(-1) != pseudo_label_all.unsqueeze(0)).float()
        mask_patch_filter = (
                (FC.unsqueeze(-1) + FC_all.unsqueeze(0)) <= (self.bdp_threshold + self.fdp_threshold)).float()
        mask_pixel_filter = torch.cat([torch.ones(mask_pixel_filter.size(0), 1).float().cuda(), mask_pixel_filter], 1)
        mask_patch_filter = torch.cat([torch.ones(mask_patch_filter.size(0), 1).float().cuda(), mask_patch_filter], 1)

        neg = torch.div(torch.matmul(anchor, neg_pair.T), self.temp)
        neg = torch.cat([pos, neg], 1)
        max = torch.max(neg, 1, keepdim=True)[0]
        exp_neg = (torch.exp(neg - max) * mask_pixel_filter * mask_patch_filter).sum(-1)

        loss = torch.exp(pos - max).squeeze(-1) / (exp_neg + self.eps)
        loss = -torch.log(loss + self.eps)

        return loss


class ConsistencyWeight(nn.Module):
    def __init__(self, max_weight, max_epoch, ramp='sigmoid'):
        super(ConsistencyWeight, self).__init__()
        self.max_weight = max_weight
        self.max_epoch = max_epoch
        self.ramp = ramp

    def forward(self, epoch):
        current = np.clip(epoch, 0.0, self.max_epoch)
        phase = 1.0 - current / self.max_epoch
        if self.ramp == 'sigmoid':
            ramps = float(np.exp(-5.0 * phase * phase))
        elif self.ramp == 'log':
            ramps = float(1 - np.exp(-5.0 * current / self.max_epoch))
        elif self.ramp == 'exp':
            ramps = float(np.exp(5.0 * (current / self.max_epoch - 1)))
        else:
            ramps = 1.0

        consistency_weight = self.max_weight * ramps
        return consistency_weight


if __name__ == '__main__':

    con = consistency_weight(1.0, 500, rampup_starts=3, rampup_ends=80, ramp_type='exp_rampup')

    for i in range(80):

        print(i, con(i, 0))
