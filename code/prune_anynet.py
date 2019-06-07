import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, transforms

import anynet
import pprint
import compute_flops

#from vgg import slimmingvgg as vgg11

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)



# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--base_model', default='resnet18', type=str,
                    help='Base model.')
parser.add_argument('--data', type=str, default='',
                    help='Path to imagenet validation data')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(os.path.dirname(args.save)):
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

pprint.pprint(args)


model = anynet.Anynet(args.base_model, num_bins=66)

#model.features = nn.DataParallel(model.features)
#cudnn.benchmark = True

#model.cuda(gpu)
#model.cuda()
model.cpu()

skdgjasdg
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        #checkpoint = torch.load(args.model)
        saved_state_dict = torch.load(args.model)
        model.load_state_dict(saved_state_dict)

        #args.start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        #model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no model found at '{}'".format(args.model))

#show original model parameters and flops
print('------------------------------')
print('Original model: ')
compute_flops.print_model_param_nums(model)

#print(model)
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre)
        mask = mask.float().cpu()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')


#torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
print('output file: {}'.format(args.save))
torch.save(model.state_dict(), args.save)

pruned_ratio = pruned/total

print('Pruned ratio: {}'.format(pruned_ratio))


print('------------------------------')
print('Pruned model: ')
compute_flops.print_model_param_nums(model)


print('Pre-processing Successful!')


