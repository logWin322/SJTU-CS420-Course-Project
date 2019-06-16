'''Train a more robust model on adversarial samples with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *


from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import LinfPGDAttack

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 300

# Data preprocessing 
# TRAINING PHASE: random crop + flip  --> 44 * 44 images
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# TESTING PHASE
transform_test = transforms.Compose([
    transforms.Resize([44, 44]),
    transforms.ToTensor(),
])


# load the dataset 
trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)



# Model based on VGG19
path = "FER2013_VGG19"
net = VGG('VGG19')


adversary = LinfPGDAttack(
    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

if use_cuda:
    net.cuda()

# 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        ori = inputs
        
        # generate adversarial samples
        with ctx_noparamgrad_and_eval(net):
            inputs = adversary.perturb(inputs, targets)
        
        # concatenate with clean samples
        inputs = torch.cat((ori, inputs), 0)
        targets = torch.cat((targets, targets), 0)

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total
    Train_loss = train_loss/len(trainloader.dataset)
    return Train_acc, Train_loss


# Private Test
def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    adv_loss = 0
    adv_correct = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        PrivateTest_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
        # generate adversarial samples 
        adv_data = adversary.perturb(inputs, targets)

        with torch.no_grad():
            adv_outputs = net(adv_data)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.data[0]
            pred = adv_outputs.max(1, keepdim=True)[1]
            adv_correct += pred.eq(targets.view_as(pred)).sum().item()

    adv_loss /= len(PrivateTestloader.dataset)
    # Save checkpoint.
    PrivateTest_acc = 100.*adv_correct/total
    print('\nTest set: avg cln loss: {:.4f},'
            ' cln acc: {}/{} ({:.0f}%)\n'.format(
                PrivateTest_loss/(batch_idx + 1), correct, total,
                100. * correct / total))
    print('Test set: avg adv loss: {:.4f},'
            ' adv acc: {}/{} ({:.0f}%)\n'.format(
                adv_loss, adv_correct, total,
                100. * adv_correct / total))
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model_adv.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

    return PrivateTest_acc, PrivateTest_loss

Train_acc_lst = []
Train_loss_lst = []
PrivateTest_acc_lst = []
PrivateTest_loss_lst = []

for epoch in range(start_epoch, total_epoch):
    train_acc, train_loss = train(epoch)
    #PublicTest(epoch)
    test_acc, test_loss = PrivateTest(epoch)
    PrivateTest_acc_lst.append(test_acc)
    PrivateTest_loss_lst.append(test_loss)
    Train_acc_lst.append(train_acc)
    Train_loss_lst.append(train_loss)
# Saving loss and accuracy
np.save(os.path.join(opt.model+"_"+"Results", "test_advacc.npy"), PrivateTest_acc_lst)
np.save(os.path.join(opt.model+"_"+"Results", "test_advloss.npy"), PrivateTest_loss_lst)
np.save(os.path.join(opt.model+"_"+"Results", "train_advacc.npy"), Train_acc_lst)
np.save(os.path.join(opt.model+"_"+"Results", "train_advloss.npy"), Train_loss_lst)
print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

