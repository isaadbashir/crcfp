import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.distributions.uniform import Uniform


class DropOutDecoder(nn.Module):
    def __init__(self, num_classes, drop_rate=0.1, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.classifier = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        x = self.classifier(self.dropout(x))
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.classifier = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.75, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.classifier(self.feature_dropout(x))
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)
        self.classifier = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.classifier(self.feature_based_noise(x))
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.classifier = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        r_adv = get_r_adv(x, self.classifier, self.it, self.xi, self.eps)
        x = self.classifier(x + r_adv)
        return x
