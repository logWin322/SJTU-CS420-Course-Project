'''Attack on sampled images from FER2013 '''
# Sample 5 images and apply the attacking method
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

import matplotlib.pyplot as plt

from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

import os
import argparse
import torch
import torch.nn as nn

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, MomentumIterativeAttack

# Seven Emotions
emotion_lst = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Sample 5 images to apply attacks
batch_size = 5

# Using the trained VGG19 model
path = "FER2013_VGG19"
model = VGG("VGG19")
checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))
model.load_state_dict(checkpoint['net'])
model.to(device)


# Preprocessing the images, sample from training set
cut_size = 44
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# sample the 23th batch data
index = 0
for cln_data, true_label in trainloader:
    index += 1
    if index == 23:
        break

bs, c, h, w = np.shape(cln_data)
# print true labels
print(true_label)
cln_data = cln_data.view(-1, c, h, w)
cln_data, true_label = cln_data.to(device), true_label.to(device)


# MomentumIterativateAttack
adversary = MomentumIterativeAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2,
    nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    targeted=False)



# L2PGDAttack
'''
adversary = L2PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)
'''

# LinfPGDAttack
'''
adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)
'''


# generate untargeted adversarial samples
adv_untargeted = adversary.perturb(cln_data, true_label)


# generate targeted adversarial samples
target = torch.ones_like(true_label) * 3
adversary.targeted = True
adv_targeted = adversary.perturb(cln_data, target)

# Test the model on these samples
pred_cln = predict_from_logits(model(cln_data))
pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
pred_targeted_adv = predict_from_logits(model(adv_targeted))

# Show the results
# Model performacne on clean, untargeted and targeted images
# ----------------------------------------------------------
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(emotion_lst[pred_cln[ii]]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("untargeted \n adv \n pred: {}".format(
        emotion_lst[pred_untargeted_adv[ii]]))
    plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
    _imshow(adv_targeted[ii])
    plt.title("targeted to 3 \n adv \n pred: {}".format(
        emotion_lst[pred_targeted_adv[ii]]))

plt.tight_layout()
plt.savefig("gen_adv.png")
# ---------------------------------------------------------

# defending based on preprocessing (feature squeezing)
# -------------------------------------------------
bits_squeezing = BitSqueezing(bit_depth=5)
median_filter = MedianSmoothing2D(kernel_size=3)
jpeg_filter = JPEGFilter(10)

defense = nn.Sequential(
    jpeg_filter,
    bits_squeezing,
    median_filter,
)

# defemse on adversarial samples and clean samples
adv = adv_untargeted
adv_defended = defense(adv)
cln_defended = defense(cln_data)

# Test the model on these samples
pred_cln = predict_from_logits(model(cln_data))
pred_cln_defended = predict_from_logits(model(cln_defended))
pred_adv = predict_from_logits(model(adv))
pred_adv_defended = predict_from_logits(model(adv_defended))


plt.figure(figsize=(10, 10))
for ii in range(batch_size):
    plt.subplot(4, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size)
    _imshow(cln_data[ii])
    plt.title("defended clean \n pred: {}".format(pred_cln_defended[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size * 2)
    _imshow(adv[ii])
    plt.title("adv \n pred: {}".format(
        pred_adv[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size * 3)
    _imshow(adv_defended[ii])
    plt.title("defended adv \n pred: {}".format(
        pred_adv_defended[ii]))

plt.tight_layout()
plt.savefig("defend_inputs.png")
#----------------------------------------------



# Attack under different eps (maximum distortion value) 
# eps = 0.1, 0.2 and 0.3. 
# --------------------------------------------------------

# eps = 0.1
adversary = MomentumIterativeAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.10,
    nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    targeted=False)
adv_untargeted_one = adversary.perturb(cln_data, true_label)
pred_cln = predict_from_logits(model(cln_data))
pred_untargeted_adv_one = predict_from_logits(model(adv_untargeted_one))

# eps = 0.2
adversary = MomentumIterativeAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.20,
    nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    targeted=False)
adv_untargeted_two = adversary.perturb(cln_data, true_label)
pred_untargeted_adv_two = predict_from_logits(model(adv_untargeted_two))

# eps = 0.3
adversary = MomentumIterativeAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.30,
    nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    targeted=False)
adv_untargeted_three = adversary.perturb(cln_data, true_label)
pred_untargeted_adv_three = predict_from_logits(model(adv_untargeted_three))

# show the results
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(adv_untargeted_one[ii])
    plt.title("eps = 0.10 \n adv \n pred: {}".format(
        emotion_lst[pred_untargeted_adv_one[ii]]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted_two[ii])
    plt.title("eps = 0.20 \n adv \n pred: {}".format(
        emotion_lst[pred_untargeted_adv_two[ii]]))
    plt.subplot(3, batch_size, ii + 1 + batch_size*2)
    _imshow(adv_untargeted_three[ii])
    plt.title("eps = 0.30 \n pred: {}".format(emotion_lst[pred_untargeted_adv_three[ii]]))
plt.tight_layout()
plt.savefig("gen_adv.png")

# ----------------------------------------------------------