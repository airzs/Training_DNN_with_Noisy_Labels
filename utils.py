# -*- coding: utf-8 -*-
# Author : zhangsen (1846182474@qq.com)
# Date : 2019-11-04
# Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

np.random.seed(0)

# Add weight decay of noisy layer to cross entropy loss
class ButtomUpDNN2Loss(nn.Module):
    """Cross Entropy with L2 penalization of confusion matrix"""
    def __init__(self, model, alpha=0.1):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, input, target):
        cross_entropy = F.cross_entropy(input, target)
        cmatrix = self.model.confusion.weight
        num_weight = cmatrix.shape[0] * cmatrix.shape[1]
        penalization = torch.sum(torch.pow(cmatrix, 2)) / (2 * num_weight)

        loss = cross_entropy + self.alpha * penalization
        return loss


class NLNNLoss(nn.Module):
    """A soft version of cross-entropy loss"""
    def __init__(self):
        super().__init__()

    def forward(self, input, estimate):
        """

        Args:
            input: num_example * num_class
            estimate: num_example * num_class

        Returns:

        """
        num_example, _ = input.shape
        log_softmax = F.log_softmax(torch.Tensor(input), dim=1)
        results = log_softmax.mul(torch.Tensor(estimate))

        # estimate_label = torch.argmax(estimate, dim=1, keepdim=True)
        # onehot = torch.zeros(input.shape).scatter_(1, estimate_label, 1)
        # results = log_softmax.mul(onehot)

        return - torch.mean(results)


# Setting random seed
def setup_seed(seed=0):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True


# Q matrix used to synthesis noisy data
def Q_matrix(n, base_prob=0.4):
    identity = base_prob * np.identity(n)
    noise = np.random.random([n, n])
    np.fill_diagonal(noise, 0.)
    noise = (1 - base_prob) * noise / noise.sum(axis=1, keepdims=1)
    qmatrix = identity + noise

    # Save Q matrix
    plt.matshow(qmatrix)
    plt.colorbar()
    plt.savefig("./imgs/true_Q.png")
    return qmatrix


# Compute confusion matrix based on true label and predicted label
def confusion_matrix(y_true, y_pred):
    cmatrix = torch.Tensor(metrics.confusion_matrix(y_true, y_pred))
    cmatrix = cmatrix / cmatrix.sum(dim=1, keepdim=True)
    return cmatrix


# Adjust learning rate of optimizer
def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
