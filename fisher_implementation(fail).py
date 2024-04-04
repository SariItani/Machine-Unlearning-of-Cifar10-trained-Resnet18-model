'''Implement Fisher Masking on CIFAR10 with PyTorch - Resnet18.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def compute_fisher(model, dataloader, device):
    model.train()
    fisher_information = {}

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in fisher_information:
                    fisher_information[name] = param.grad.detach().clone() ** 2
                else:
                    fisher_information[name] += param.grad.detach().clone() ** 2

    for name in fisher_information:
        fisher_information[name] /= len(dataloader)

    return fisher_information


def filter_forget_class(data, targets, forget_class):
    forget_mask = [label != classes.index(forget_class) for label in targets]
    filtered_data = [img for img, mask in zip(data, forget_mask) if mask]
    filtered_targets = [label for label,
                        mask in zip(targets, forget_mask) if mask]
    return filtered_data, filtered_targets


threshold = 1e-4


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_forget = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_retain = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Choose the class to forget
forget_class = 'bird'

# Modify the training dataset to create forget and retain sets
forget_set_data, forget_set_targets = filter_forget_class(
    trainset.data, trainset.targets, forget_class)
retain_set_data, retain_set_targets = filter_forget_class(
    trainset.data, trainset.targets, forget_class)

# Convert targets to integers (assuming they are initially numpy arrays)
forget_set_targets = np.array(forget_set_targets).astype(int)
retain_set_targets = np.array(retain_set_targets).astype(int)

# Create datasets for forget and retain sets
forget_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_forget
)
forget_set.data = forget_set_data
forget_set.targets = forget_set_targets

retain_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_retain
)
retain_set.data = retain_set_data
retain_set.targets = retain_set_targets

# Create dataloaders for forget and retain sets
forget_loader = torch.utils.data.DataLoader(
    forget_set, batch_size=128, shuffle=True, num_workers=2
)
retain_loader = torch.utils.data.DataLoader(
    retain_set, batch_size=128, shuffle=True, num_workers=2
)


# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train_forget(epoch, forget_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # Calculate Fisher Information
    fisher_information = compute_fisher(net, retain_loader, device)

    for batch_idx, (inputs, targets) in enumerate(forget_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero-out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Fisher unlearning: mask parameters with low Fisher information
        for name, param in net.named_parameters():
            if param.requires_grad:
                if name in fisher_information:
                    mask = (fisher_information[name] > threshold).float()
                    param.grad *= mask

        # Update weights
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(forget_loader), 'Forget Loss: %.3f | Forget Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def train_retain(epoch, retain_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(retain_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero-out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(retain_loader), 'Retain Loss: %.3f | Retain Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test_forget(epoch, forget_loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(forget_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(forget_loader), 'Forget Test Loss: %.3f | Forget Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test_retain(epoch, retain_loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(retain_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(retain_loader), 'Retain Test Loss: %.3f | Retain Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(start_epoch, start_epoch + 100):
    train_forget(epoch, forget_loader)
    test_forget(epoch, forget_loader)
    train_retain(epoch, retain_loader)
    test_retain(epoch, retain_loader)
    scheduler.step()
