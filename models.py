# -*- coding: utf-8 -*-
# Author : zhangsen (1846182474@qq.com)
# Date : 2019-11-04
# Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseDNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.noisy_model = False
        self.mode = "BaseModel"
        self.params = params
        self.fc1 = nn.Linear(params["in"], params["fc1"])
        self.fc2 = nn.Linear(params["fc1"], params["fc2"])
        self.fc3 = nn.Linear(params["fc2"], params["out"])

    def forward(self, x):
        x = x = x.view(-1, self.params["in"])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ButtomUpDNN1(BaseDNN):
    def __init__(self, params):
        super().__init__(params)
        self.noisy_model = True
        # "BaseModel" or "NoisyModel" to determine whether to use noisy layer
        self.mode = "BaseModel"
        self.confusion = nn.Linear(self.params["out"], self.params["out"], bias=False)
        self.confusion.weight.requires_grad = False

    def forward(self, x):
        x = super().forward(x)
        if self.mode == "NoisyModel":
            x = F.softmax(x, dim=1)
            x = self.confusion(x)

        return x


class ButtomUpDNN2(BaseDNN):
    def __init__(self, params):
        super().__init__(params)
        self.noisy_model = True
        # "BaseModel" or "NoisyModel" to determine whether to use noisy layer
        self.mode = "BaseModel"
        self.confusion = nn.Linear(self.params["out"], self.params["out"], bias=False)
        self.confusion.weight = nn.Parameter(torch.eye(self.params["out"]))

    def forward(self, x):
        x = super().forward(x)
        if self.mode == "NoisyModel":
            x = F.softmax(x, dim=1)
            x = self.confusion(x)
        return x


class NLNN(BaseDNN):
    def __init__(self, params):
        super().__init__(params)
        self.noisy_model = True
        # "BaseModel" or "NoisyModel" to determine whether to use noisy layer
        self.mode = "BaseModel"
        self.confusion = torch.zeros(self.params["out"], self.params["out"])
