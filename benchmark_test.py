''' STRICTLY FOR TEST/INFERENCE '''
''' DONOT EVEN USE FOR PARTIAL COMPUTE '''

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
parser.add_argument('--arch', help="which architecture to use", required=True)
parser.add_argument('--iters', default=100, type=int, help='iters')
parser.add_argument('--batch_size', default=64, type=int, help='iters')


# CUDNN Enabled
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True

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
print('==> Building model..')
if args.arch == "vgg19":
    net = VGG('VGG19')
elif args.arch == "vgg16":
    net = VGG('VGG16')
elif args.arch == "vgg11":
    net = VGG('VGG11')
elif args.arch == "resnet152":
    net = ResNet152()
elif args.arch == "resnet18":
    net = ResNet18()
elif args.arch == "resnet50":
    net = ResNet50()    
elif args.arch == "resnet34":
    net = ResNet34()
elif args.arch == "resnet101":
    net = ResNet101()
elif args.arch == "googlenet":
    net = GoogLeNet()
elif args.arch == "alexnet":
    raise NotImplementedError

net = net.to(device)
model_mem_usage = torch.cuda.memory_allocated()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# training time, forward and backward 
f_test = []
peak_memory_usage = []
memory_usage = []
input_memory_usage = []

net.eval()
#with torch.no_grad():
if 1:
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx > args.iters:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        input_memory_usage =  [torch.cuda.memory_allocated()]

        f_start = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)

        # forward pass
        f_start.record()
        outputs = net(inputs)
        f_end.record()
        torch.cuda.synchronize()
        f_time = f_start.elapsed_time(f_end)
        f_test += [f_time]
        memory_usage += [torch.cuda.memory_allocated()]
        peak_memory_usage += [torch.cuda.max_memory_allocated()]



out = ("{},{},{},{},{},{},{}".format(args.arch, model_mem_usage, args.batch_size, statistics.mean(f_test), statistics.median(f_test), max(f_test), min(f_test))) 
out += "," + "{},{},{}".format(statistics.mean(peak_memory_usage), statistics.median(memory_usage), statistics.median(input_memory_usage))

with open("test_benchmark_fullmodel_wo_nograd.txt", "a") as writer:
    writer.write(out+"\n")
 
