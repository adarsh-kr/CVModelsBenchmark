'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import statistics

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--arch', help="which architecture to use")
parser.add_argument('--iters', default=20, type=int, help='iters')

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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.arch == "vgg19":
    net = VGG('vgg19')
elif args.arch == "vgg16":
    net = VGG('vgg16')
elif args.arch == "vgg11":
    net = VGG('vgg11')
elif args.arch == "resnet152":
    net = ResNet152()
elif args.arch == "resnet18":
    net = ResNet18()
elif args.arch == "resnet50":
    net = ResNet50()    
elif args.arch == "resnet101":
    net = ResNet101()
elif args.arch == "google":
    net = GoogLeNet()
elif args.arch == "alexnet":
    raise NotImplementedError

# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# 

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# training time, forward and backward 
f_train = []
b_train = []
peak_memory_usage = []

net.train()
for i in range(args.iters):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        f_start = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)

        # forward pass
        f_start.record()
        optimizer.zero_grad()
        outputs = net(inputs)
        f_end.record()
        torch.cuda.synchronize()
        f_time = f_start.elapsed_time(f_end)
        f_train += [f_time]

        b_start = torch.cuda.Event(enable_timing=True)
        b_end   = torch.cuda.Event(enable_timing=True)
        # backward pass
        b_start.record()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        b_end.record()
        b_time = b_start.elapsed_time(b_end)
        b_train += [b_time]


        torch.cuda.synchronize()
        peak_memory_usage += [torch.cuda.reset_max_memory_allocated()]


print("Forward Time")
print(f_time)
print("Avg: {}, Max: {}, Min: {}".format(statistics.mean(f_time), statistics.max(f_time), statistics.min(f_time)))
print(b_time)
print("Avg: {}, Max: {}, Min: {}".format(statistics.mean(b_time), statistics.max(b_time), statistics.min(b_time)))
print(peak_memory_usage)
print("Avg: {}, Max: {}, Min: {}".format(statistics.mean(peak_memory_usage), statistics.max(peak_memory_usage), statistics.min(peak_memory_usage)))
