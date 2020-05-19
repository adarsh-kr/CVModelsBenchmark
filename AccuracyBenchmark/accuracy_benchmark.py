#!/usr/bin/env python

import os, argparse, time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time, random
from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from tqdm import trange

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_model(arch):
    if arch == "resnet152":
        net = ResNet152()
    elif arch == "resnet18":
        net = ResNet18()
    elif arch == "resnet50":
        net = ResNet50()    
    elif arch == "resnet34":
        net = ResNet34()
    elif arch == "resnet101":
        net = ResNet101()
    elif arch  == "vgg16":
        net = VGG('VGG16')
    elif arch  == "vgg11":
        net = VGG('VGG11')
    elif arch  == "vgg19":
        net = VGG('VGG19')
    elif arch == "googlenet":
        net = GoogLeNet()
    elif arch == "dense121":
        net = DenseNet121()
    elif arch == "desne201":
        net = DenseNet201()
    elif arch == "lenet":
        net = LeNet()
    elif arch == "mobile":
        net = MobileNet()
    else:
        raise NotImplementedError
    return net

class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_CIFAR(batch_size, args):
    dataset = datasets.CIFAR10('./data_cifar', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ]))

    test_dataset = datasets.CIFAR10('./data_cifar', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ]))

    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    if args.shard_data:
        partition_sizes = [1.0 / size for _ in range(size)]
    else:
        partition_sizes = [1.0]

    # print(partition_sizes)
    partition = DataPartitioner(dataset, partition_sizes)
    if args.shard_data:
        partition = partition.use(dist.get_rank())
    else:
        partition = partition.use(0)
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    test_set = torch.utils.data.DataLoader(test_dataset, batch_size=bsz, shuffle=False)
    return train_set, test_set, bsz


""" Gradient averaging. """
def average_gradients(model, args):
    size = float(dist.get_world_size())
    bucket_size = len(list(model.named_parameters()))/size

    for idx, param in enumerate(model.parameters()):
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        if args.reduce_op == "avg":
            param.grad.data /= size
        elif args.reduce_op == "avg_w_count":
            num_wrkrs_sending_update = idx/bucket_size  
            param.grad.data /= (num_wrkrs_sending_update+1)

        

""" Distributed Synchronous SGD Example """
def run(rank, size, args):
    device = torch.device('cuda:{}'.format(rank%torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(1234)
    train_set, test_set, bsz = partition_CIFAR(args.batch_size, args)
    #  
    model = get_model(args.arch)
    model.to(device)
    
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(),
                            lr=0.01, momentum=0.5)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(),
                    lr=0.001)
    
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    
    dropping_scheme_name = ["no_drop", "jgsaw"]
    create_dir(args.arch)

    file = open("{}/scheme_{}_workers_{}_updateFreq_{}_totalBsz_{}_optim_{}_shard_{}_reduceOP_{}_start_epoch{}_cyclic_{}.log".format(args.arch, args.dropping_scheme, size, args.update_granularity, args.batch_size, args.optim, args.shard_data, args.reduce_op, args.start_epoch, args.cyclic), "w", buffering=1)
    file.write(",".join(["Rank", "Epoch", "TrainLoss", "TrainAcc", "TestLoss", "TestAcc"]) + "\n")
    
    for epoch in range(total_epochs):
        model.train()
        train_acc = Accuracy()
        train_loss = Average()

        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_set):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                # data = data.cuda()
                # target = target.cuda()
                data = data.to(device)
                target = target.to(device)
            
            output = model(data)

            loss = F.cross_entropy(output, target)
            epoch_loss += loss.item()
            loss.backward()
            
            if batch_idx%100==0:
                print("Rank:{}, Epoch:{}, Batch_Idx:{}, Loss:{}".format(rank, epoch, batch_idx, loss.item()))
            
            train_acc.update(output, target)
            train_loss.update(loss.item(), data.size(0))
            
            if epoch >= args.start_epoch:
                if args.dropping_scheme == "no_drop":
                    # standard training
                    pass   
                
                elif args.dropping_scheme == "jgsaw":
                    with torch.no_grad():
                        # gradient send for suffix network 
                        rsz = Random()
                        rsz.seed(batch_idx)

                        buckets = []
                        bucket_size = len(list(model.named_parameters()))/size
                        for i in range(size):
                            buckets += [i*int(bucket_size)]

                        # add args.update_granularity: scheme is changed after every update_granularity iterations  
                        if args.cyclic==True:
                            is_bucket = (rank + int(batch_idx/args.update_granularity))%size
                        else:
                            is_bucket = rank
                            rsz.shuffle(buckets)
                        
                        layer_segment = buckets[is_bucket]
                        count = 0
                        for layer_idx, (name, param) in enumerate(model.named_parameters()):
                            if  layer_segment <=layer_idx:
                                count += 1
                                pass 
                            else:
                                # drop the gradient
                                param.grad = F.dropout(param.grad, 1.0)
                    
            average_gradients(model, args)
            optimizer.step()
        model.eval()
        test_acc = Accuracy()
        test_loss = Average()
        for test_data, test_tgt in test_set:
            if torch.cuda.is_available():
                # test_data = test_data.cuda()
                # test_tgt = test_tgt.cuda()
                test_data = test_data.to(device)
                test_tgt = test_tgt.to(device)
            output = model(test_data)
            loss = F.nll_loss(output, test_tgt)
            test_loss.update(loss.item(), test_data.size(0))
            test_acc.update(output, test_tgt)

        if dist.get_rank()==0:
            print('Rank ', dist.get_rank(), ', epoch ',
                epoch, 'train_loss', train_loss.average, 'train_acc', train_acc.accuracy, 'test_loss', test_loss.average, 'test_acc', test_acc.accuracy)
            _ = [str(x) for x in [dist.get_rank(), epoch,  train_loss.average, train_acc.accuracy, test_loss.average, test_acc.accuracy]]
            file.write(",".join(_) + "\n")

def init_processes(rank, size, args, fn, port=29534, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_prcs', type=int, default=2, help='num process')
    parser.add_argument('--epochs', type=int, default=40,  help='total epochs')
    parser.add_argument('-bsz', '--batch_size', type=int, default=128,  help='total epochs')
    parser.add_argument('--dropping_scheme', type=str, default='jgsaw',  help='dropping scheme to follow')
    parser.add_argument('--port', type=int, default=29534,  help=' master port')
    parser.add_argument('--update_granularity', type=int, default=1, help="how frequently to update the dropping scheme")
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--shard_data', action='store_true')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--reduce_op', type=str, default='avg', help="avg_w_count")
    parser.add_argument('--start_epoch', type=int, default=0, help="after this epoch any scheme starts")
    parser.add_argument('--cyclic', action='store_true')

    args = parser.parse_args()
    print(args)
    time.sleep(5)    
    size = args.num_prcs
    total_epochs = args.epochs
    dropping_scheme = args.dropping_scheme
    port = args.port

    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, args, run, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
