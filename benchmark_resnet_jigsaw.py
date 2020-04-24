'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, time
import argparse

from models import *
from utils import progress_bar
import statistics

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--arch', help="which architecture to use")
parser.add_argument('--iters', default=100, type=int, help='iters')
parser.add_argument('--batch_size', default=64, type=int, help='iters')
parser.add_argument('--n', default=0, type=int, help='first n not doing grad')

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.arch == "resnet152":
    net = ResNet152()
elif args.arch == "resnet18":
    net = ResNet18()
elif args.arch == "resnet50":
    net = ResNet50()    
elif args.arch == "resnet34":
    net = ResNet34()
elif args.arch == "resnet101":
    net = ResNet101()


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
memory_usage = []
net.train()
# just to warmup the gpu
#for batch_idx, (inputs, targets) in enumerate(trainloader):
#    if batch_idx > args.iters:
#        break
#    inputs = inputs.to(device)
#    net(inputs)


net.train()
for batch_idx, (inputs, targets) in enumerate(trainloader):
    if batch_idx > args.iters:
        break
    inputs, targets = inputs.to(device), targets.to(device)
    f_start = torch.cuda.Event(enable_timing=True)
    f_end = torch.cuda.Event(enable_timing=True)

    # forward pass
    f_start.record()
    outputs = net(inputs, args.n)
    f_end.record()
    torch.cuda.synchronize()
    f_time = f_start.elapsed_time(f_end)
    f_train += [f_time]

    b_start = torch.cuda.Event(enable_timing=True)
    b_end   = torch.cuda.Event(enable_timing=True)
    # backward pass
    b_start.record()
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    b_end.record()
    torch.cuda.synchronize()
    b_time = b_start.elapsed_time(b_end)
    b_train += [b_time]

    memory_usage += [torch.cuda.memory_allocated()]
    peak_memory_usage += [torch.cuda.max_memory_allocated()]


#print("Forward Time")
#print(f_train)
#print("{},{},{},{},{}".format(args.arch,statistics.mean(f_train), statistics.median(f_train),  max(f_train), min(f_train)))
#print(b_train)
#print("{},{},{},{},{}".format(args.arch,statistics.mean(b_train), statistics.median(b_train), max(b_train), min(b_train)))
#print(peak_memory_usage)
#print("{},{},{},{},{}".format(args.arch,statistics.mean(peak_memory_usage), statistics.median(peak_memory_usage), max(peak_memory_usage), min(peak_memory_usage)))


out = ("{},{},{},{},{},{}".format(args.arch,args.batch_size,statistics.mean(f_train), statistics.median(f_train),  max(f_train), min(f_train))) 
out += "," + ("{},{},{},{}".format(statistics.mean(b_train), statistics.median(b_train),  max(b_train), min(b_train)))
out += "," + "{},{},{},{},{}".format(statistics.mean(peak_memory_usage), statistics.median(memory_usage), max(memory_usage), min(memory_usage), args.n)

with open("benchmark_resnet_jigsaw.txt", "a") as writer:
    writer.write(out+"\n")
 
