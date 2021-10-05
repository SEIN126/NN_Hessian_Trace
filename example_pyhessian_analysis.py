#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils import *
from hessian import hessian

from pytorchcv.model_provider import get_model as ptcv_get_model # model

# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

model = ptcv_get_model("resnet20_cifar10", pretrained=True)
# change the model to eval mode to disable running stats upate
model.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData()

# for illustrate, we only use one batch to do the tutorial
for inputs, targets in train_loader:
    break

# we use cuda to make the computation fast
model = model.cuda()
inputs, targets = inputs.cuda(), targets.cuda()

# get hessian trace
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
avg_dict_trace_layer, avg_dict_trace_layer_origin = hessian_comp.trace(maxIter=150)

# check the relative error
print(sum(avg_dict_trace_layer.values()))
print(sum(avg_dict_trace_layer_origin.values()))
relative_error(avg_dict_trace_layer, avg_dict_trace_layer_origin)

# make hessian trace list
trace_origin = [x for x in avg_dict_trace_layer_origin.values()]
trace = [x for x in avg_dict_trace_layer_origin.values()]

# plot and save the graph
plt.plot(trace_origin, label = 'origin')
plt.plot(trace, label = 'check')
plt.legend(loc = 'upper right')
plt.show()
plt.savefig('./resnet20_cifar10_example.png')

