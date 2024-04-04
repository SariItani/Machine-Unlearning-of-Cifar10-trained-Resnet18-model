'''Implement Fisher Masking on CIFAR10 with PyTorch - Resnet18.'''
import numpy as np
from sklearn import linear_model, model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
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
    trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# defining the noise structure
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


# list of all classes
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# classes which are required to un-learn
classes_to_forget = [0, 2]

# classwise list of samples
num_classes = 10
classwise_train = {}
for i in range(num_classes):
    classwise_train[i] = []

for img, label in trainset:
    classwise_train[label].append((img, label))

classwise_test = {}
for i in range(num_classes):
    classwise_test[i] = []

for img, label in testset:
    classwise_test[label].append((img, label))

# getting some samples from retain classes
num_samples_per_class = 1000

retain_samples = []
for i in range(len(classes)):
    if classes[i] not in classes_to_forget:
        retain_samples += classwise_train[i][:num_samples_per_class]

# retain validation set
retain_valid = []
for cls in range(num_classes):
    if cls not in classes_to_forget:
        for img, label in classwise_test[cls]:
            retain_valid.append((img, label))

# forget validation set
forget_valid = []
for cls in range(num_classes):
    if cls in classes_to_forget:
        for img, label in classwise_test[cls]:
            forget_valid.append((img, label))

batch_size = 128
forget_valid_dl = DataLoader(
    forget_valid, batch_size, num_workers=3, pin_memory=True)
retain_valid_dl = DataLoader(
    retain_valid, batch_size*2, num_workers=3, pin_memory=True)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    checkpoint = torch.load('./checkpoint/forget_chkpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Performance after Impair Step
noises = {}
# for cls in classes_to_forget:
#     print("Optiming loss for class {}".format(cls))
#     noises[cls] = Noise(batch_size, 3, 32, 32)
#     opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)

#     num_epochs = 5
#     num_steps = 8
#     class_label = cls
#     for epoch in range(num_epochs):
#         total_loss = []
#         for batch in range(num_steps):
#             inputs = noises[cls]()
#             labels = torch.zeros(batch_size)+class_label
#             outputs = net(inputs)
#             loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * \
#                 torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             total_loss.append(loss.cpu().detach().numpy())
#         print("Loss: {}".format(np.mean(total_loss)))

for cls in classes_to_forget:
    print("Optimizing loss for class {}".format(cls))
    noises[cls] = Noise(batch_size, 3, 32, 32)
    opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)

    num_epochs = 5
    num_steps = 8
    class_label = cls
    for epoch in range(num_epochs):
        total_loss = []
        for batch in range(num_steps):
            inputs = noises[cls]()
            labels = torch.zeros(batch_size) + class_label
            outputs = net(inputs)
            # Use the correct class for cross-entropy loss
            loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * \
                torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())
        print("Loss: {}".format(np.mean(total_loss)))

noisy_data = []
num_batches = 20
class_num = 0

for cls in classes_to_forget:
    for i in range(num_batches):
        batch = noises[cls]().cpu().detach()
        for i in range(batch[0].size(0)):
            noisy_data.append((batch[i], torch.tensor(class_num)))

other_samples = []
for i in range(len(retain_samples)):
    other_samples.append(
        (retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))
noisy_data += other_samples
noisy_loader = torch.utils.data.DataLoader(
    noisy_data, batch_size=128, shuffle=True)


optimizer = torch.optim.Adam(net.parameters(), lr=0.02)


for epoch in range(1):
    net.train(True)
    running_loss = 0.0
    running_acc = 0
    for i, data in enumerate(noisy_loader):
        inputs, labels = data
        inputs, labels = inputs, torch.tensor(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() * inputs.size(0)
        out = torch.argmax(outputs.detach(), dim=1)
        assert out.shape == labels.shape
        running_acc += (labels == out).sum().item()
    print(
        f"Train loss {epoch+1}: {running_loss/len(trainset)},Train Acc:{running_acc*100/len(trainset)}%")


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def validation_step(model, batch):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return {'Loss': loss.detach(), 'Acc': acc}


def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)


print("Performance of Standard Forget Model on Forget Class")
history = [evaluate(net, forget_valid_dl)]
print("Accuracy: {}".format(history[0]["Acc"]*100))
print("Loss: {}".format(history[0]["Loss"]))

print("Performance of Standard Forget Model on Retain Class")
history = [evaluate(net, retain_valid_dl)]
print("Accuracy: {}".format(history[0]["Acc"]*100))
print("Loss: {}".format(history[0]["Loss"]))

# Performance after Repair Step
heal_loader = torch.utils.data.DataLoader(
    other_samples, batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


for epoch in range(1):
    net.train(True)
    running_loss = 0.0
    running_acc = 0
    for i, data in enumerate(heal_loader):
        inputs, labels = data
        inputs, labels = inputs, torch.tensor(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() * inputs.size(0)
        out = torch.argmax(outputs.detach(), dim=1)
        assert out.shape == labels.shape
        running_acc += (labels == out).sum().item()
    print(
        f"Train loss {epoch+1}: {running_loss/len(trainset)},Train Acc:{running_acc*100/len(trainset)}%")

print("Performance of Standard Forget Model on Forget Class")
history = [evaluate(net, forget_valid_dl)]
print("Accuracy: {}".format(history[0]["Acc"]*100))
print("Loss: {}".format(history[0]["Loss"]))

print("Performance of Standard Forget Model on Retain Class")
history = [evaluate(net, retain_valid_dl)]
print("Accuracy: {}".format(history[0]["Acc"]*100))
print("Loss: {}".format(history[0]["Loss"]))

# Save checkpoint.
acc = history[0]["Acc"]*100
if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/forget_chkpt.pth')
    best_acc = acc

def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


# print its accuracy on retain and forget set
# print(f"Retain set accuracy: {100.0 * accuracy(net, heal_loader):0.1f}%")
# print(f"Forget set accuracy: {100.0 * accuracy(net, noisy_loader):0.1f}%")

rt_test_losses = compute_losses(net, testloader)
rt_forget_losses = compute_losses(net, noisy_loader)
rt_samples_mia = np.concatenate((rt_test_losses, rt_forget_losses)).reshape((-1, 1))
labels_mia = [0] * len(rt_test_losses) + [1] * len(rt_forget_losses)
rt_mia_scores = simple_mia(rt_samples_mia, labels_mia)
print(f"The MIA has an accuracy of {rt_mia_scores.mean():.3f} on forgotten vs unseen images")
