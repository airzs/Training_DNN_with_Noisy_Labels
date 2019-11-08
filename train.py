# -*- coding: utf-8 -*-
# Author : zhangsen (1846182474@qq.com)
# Date : 2019-11-04
# Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
from preprocessing import get_dataloader, get_dataloader_bu1
from utils import ButtomUpDNN2Loss, NLNNLoss
from utils import adjust_lr, confusion_matrix

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

batch_size = 128


# Initialization of training process
def train_init(model_name, lr=0.01):
    # Parameters
    params = {
        "in": 784,
        "fc1": 500,
        "fc2": 300,
        "out": 10
    }
    # Model
    model = getattr(models, model_name)(params)
    # Data
    train_loader, test_loader = get_dataloader(batch_size=batch_size, noisy=True)
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Criterion
    criterion = nn.CrossEntropyLoss()

    return model, train_loader, test_loader, optimizer, criterion


# Test a model using a batch of test data
def test_batch(model, x, target):
    x, target = x.to(device), target.to(device)

    out = model(x)
    loss = F.cross_entropy(out, target)

    _, pred_label = torch.max(out.data, 1)
    total = x.data.size()[0]
    correct = (pred_label == target.data).sum().item()

    return total, correct, loss


# Testing a model using all test data
def test_model(model, test_loader, criterion, epoch):
    correct_cnt, total_cnt = 0, 0
    model.eval()
    model.mode = "BaseModel"

    for batch_idx, (x, target, _) in enumerate(test_loader):
        total, correct, loss = test_batch(model, x, target)
        total_cnt += total
        correct_cnt += correct
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            print(
                '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx + 1, loss, correct_cnt / total_cnt)
            )


# Train model using a batch of data
def train_batch(model, x, target, index, optimizer, criterion):
    x, target = x.to(device), target.to(device)
    optimizer.zero_grad()

    out = model(x)
    if model.__class__.__name__ == "NLNN" and \
            criterion.__class__.__name__ == "NLNNLoss":
        target = model.estimate_prob[index, :]

    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    # Keep noisy matrix a probability matrix of ButtomUpDNN2
    if model.__class__.__name__ == "ButtomUpDNN2":
        cmatrix = model.confusion.weight.data
        model.confusion.weight = nn.Parameter(cmatrix / cmatrix.sum(dim=1, keepdim=True))

    return out, loss


# Train a model for a whole epoch
def train_epoch(model, train_loader, test_loader, optimizer, criterion, epoch):
    # Training
    model.train()
    for batch_idx, (x, target, index) in enumerate(train_loader):
        out, loss = train_batch(model, x, target, index, optimizer, criterion)
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx + 1, loss)
            )

    # Testing
    test_model(model, test_loader, criterion, epoch)

    return model


# Get predicting probabilities of all data using trained model
def get_predict_prob(model, dataloader):
    y_noisy = []
    num_sample = dataloader.dataset.data.shape[0]
    num_class = len(dataloader.dataset.classes)
    predict_prob = torch.zeros((num_sample, num_class))

    model.mode = "BaseModel"
    model.eval()
    for idx, (x, target, index) in enumerate(dataloader):
        x, target = x.to(device), target.to(device)

        out = model(x)
        predict_prob[index, :] = out.data
        y_noisy = y_noisy + list(target.data.numpy())

    predict_prob = F.softmax(predict_prob, dim=1)

    return predict_prob


# Get predicting labels of all data using trained model
def get_predict_label(model, dataloader):
    predict_prob = get_predict_prob(model, dataloader)
    y_pred = torch.argmax(predict_prob, dim=1).int()
    y_noisy = dataloader.dataset.targets.int()

    return y_noisy, y_pred


# ===================================Base DNN with noisy data======================================
# Main function to train base DNN model with 60000 noisy data
def train_basednn_v1(epochs=10):
    print("======Starting to train base DNN model======")
    model, train_loader, test_loader, optimizer, criterion = train_init("BaseDNN", lr=0.01)

    model.mode = "BaseModel"
    for epoch in range(epochs):
        train_epoch(model, train_loader, test_loader, optimizer, criterion, epoch)


# =================================Base DNN with 50% clean data====================================
# Main function to train base DNN model with 30000 clean data
def train_basednn_v2(epochs=10):
    print("======Starting to train base DNN model======")
    model, train_loader, test_loader, optimizer, criterion = train_init("BaseDNN", lr=0.01)

    clean_loader, clean_train_loader, noisy_loader = get_dataloader_bu1()
    clean_dataset = torch.utils.data.ConcatDataset(
        [clean_loader.dataset, clean_train_loader.dataset])
    all_clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.mode = "BaseModel"
    for epoch in range(epochs):
        train_epoch(model, all_clean_loader, test_loader, optimizer, criterion, epoch)


# ===============================Buttom-Up DNN with 50% clean and 50% noisy data====================
# Compute R matrix using confusion matrices of clean noisy data
# TODO: 如何处理值为负的元素
def compute_rmatrix(cmatrix_clean, cmatrix_noisy):
    r = torch.mm(cmatrix_clean.inverse(), cmatrix_noisy)
    return r


# Compute estimated confusion matrix of new buttom-up DNN v1 model
def compute_estimate_confusion(rmatrix, y_noisy, num_class=10):
    # Statistic y_noisy
    y_prob = torch.zeros(num_class)
    for i in range(num_class):
        y_prob[i] = (y_noisy == i).sum()

    y_prob = y_prob / y_prob.sum()

    # Compute confusion matrix
    cmatrix = rmatrix * y_prob.reshape(-1, 1)
    cmatrix = cmatrix / cmatrix.sum(dim=1, keepdim=True)

    return cmatrix


# Train buttom-up DNN v1 model for a whole epoch
def train_epoch_bu1(model, clean_loader, noisy_loader, test_loader, optimizer, criterion, epoch):
    clean_iter, noisy_iter = iter(clean_loader), iter(noisy_loader)
    batch_idx = 0

    # Train
    while True:
        try:
            # Train base part of model using clean data
            model.mode = "BaseModel"
            c_x, c_target, c_index = next(clean_iter)
            out, loss = train_batch(model, c_x, c_target, c_index, optimizer, criterion)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(clean_loader):
                print(
                    '==>>> epoch: {}, batch index: {}, clean data train loss: {:.6f}'.format(
                        epoch, batch_idx + 1, loss)
                )

            # Train whole model using noisy data
            model.mode = "NoisyModel"
            n_x, n_target, n_index = next(noisy_iter)
            out, loss = train_batch(model, n_x, n_target, n_index, optimizer, criterion)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(noisy_loader):
                print(
                    '==>>> epoch: {}, batch index: {}, noisy data train loss: {:.6f}'.format(
                        epoch, batch_idx + 1, loss)
                )

            batch_idx += 1
        except StopIteration as e:
            break

    # Testing
    test_model(model, test_loader, criterion, epoch)

    return model


# Main function to train buttom-up DNN v1 model with 30000 clean data and 30000 noisy data
def train_buttomupdnn_v1(epochs=40, init_epochs=10, lr=0.01):
    print("======Train base DNN using clean data======")
    model, train_loader, test_loader, optimizer, criterion = train_init("BaseDNN", lr=0.05)
    clean_loader, clean_train_loader, noisy_loader = get_dataloader_bu1()
    model.mode = "BaseModel"

    # Train base DNN model
    for epoch in range(init_epochs):
        train_epoch(model, clean_train_loader, test_loader, optimizer, criterion, epoch)

    # Estimate confusion matrix for new buttom-up DNN v1 model
    y_clean, y_pred = get_predict_label(model, clean_loader)
    cmatrix_clean = confusion_matrix(y_clean, y_pred)
    y_noisy, y_pred = get_predict_label(model, noisy_loader)
    cmatrix_noisy = confusion_matrix(y_noisy, y_pred)

    rmatrix = compute_rmatrix(cmatrix_clean, cmatrix_noisy)
    cmatrix = compute_estimate_confusion(rmatrix, y_noisy)

    # Initialize new buttom-up DNN v1 model
    new_model = models.ButtomUpDNN1(model.params)
    new_model.confusion.weight = nn.Parameter(cmatrix)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Concatenate all clean dataset and get data loader
    clean_dataset = torch.utils.data.ConcatDataset(
        [clean_loader.dataset, clean_train_loader.dataset])
    all_clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Train buttom-up DNN v2 model
    for epoch in range(init_epochs, epochs):
        train_epoch_bu1(model, all_clean_loader, noisy_loader, test_loader,
                        optimizer, criterion, epoch)


# ======================================Buttom-Up DNN v2 with noisy data only==============================
# Main function to train buttom-up DNN v2 model with 60000 noisy data
def train_buttomupdnn_v2(epochs=40, init_epochs=10):
    model, train_loader, test_loader, optimizer, criterion = train_init("ButtomUpDNN2", lr=0.05)
    # Initialize base model
    print("======Initialize base DNN model======")
    model.mode = "BaseModel"
    model.confusion.weight.requires_grad = False

    for epoch in range(init_epochs):
        train_epoch(model, train_loader, test_loader, optimizer, criterion, epoch)

    # Update noisy layer
    print("======Starting to update noisy layer of ButtomUpDNN2=====")
    adjust_lr(optimizer, 0.01)
    model.mode = "NoisyModel"
    model.confusion.weight.requires_grad = True
    criterion = ButtomUpDNN2Loss(model, alpha=0.05)

    for epoch in range(init_epochs, epochs):
        train_epoch(model, train_loader, test_loader, optimizer, criterion, epoch)


# ========================================NLNN with noisy data only===================================
# Estimate hidden true labels with current parameters of model
def e_step(model, dataloader):
    predict_prob = get_predict_prob(model, dataloader)
    y_noisy = list(dataloader.dataset.targets.int().numpy())
    theta = model.confusion
    num_sample = len(y_noisy)
    for i in range(num_sample):
        label = y_noisy[i]
        predict_prob[i, :] = predict_prob[i, :] * theta[:, label].flatten()

    estimate_prob = predict_prob / torch.sum(predict_prob, dim=1, keepdim=True)
    model.estimate_prob = estimate_prob

    return estimate_prob, predict_prob


# Estimate parameters of model with current estimated hidden true labels
def m_step(model, train_loader, test_loader, estimate_prob, epoch, epochs=1, lr=0.03):
    theta = model.confusion
    num_class = estimate_prob.shape[1]
    label = train_loader.dataset.targets.int()

    # Update theta
    for i in range(num_class):
        summation = torch.sum(estimate_prob[:, i])
        for j in range(num_class):
            theta[i, j] = torch.dot(estimate_prob[:, i], (label == j).float()) / summation

    # Update parameters of DNN
    model.mode = "BaseModel"
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = NLNNLoss()
    for epoch in range(epoch, epoch + epochs):
        train_epoch(model, train_loader, test_loader, optimizer, criterion, epoch)


# Main function to train buttom-up DNN v2 model with 60000 noisy data
def train_nlnn(epochs=40, init_epochs=10):
    model, train_loader, test_loader, optimizer, criterion = train_init("NLNN", lr=0.01)
    # Initialize base model
    print("======Initialize base DNN model======")
    model.mode = "BaseModel"
    for epoch in range(init_epochs):
        train_epoch(model, train_loader, test_loader, optimizer, criterion, epoch)

    print("=======Initialize theta matrix=======")
    # Initialize the Theta matrix
    y_noisy, y_pred = get_predict_label(model, train_loader)
    model.confusion = confusion_matrix(y_noisy, y_pred)

    print("======Starting to EM steps======")
    for epoch in range(init_epochs, epochs):
        estimate_prob, predict_prob = e_step(model, train_loader)
        m_step(model, train_loader, test_loader, estimate_prob, epoch)


if __name__ == "__main__":
    # train_basednn_v1(epochs=30)
    # train_basednn_v2(epochs=30)
    train_buttomupdnn_v1(epochs=30, init_epochs=10)
    # train_buttomupdnn_v2(epochs=30, init_epochs=10)
    # train_nlnn(epochs=2, init_epochs=1)
