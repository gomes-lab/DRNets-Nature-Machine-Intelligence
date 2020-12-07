import numpy as np
import os, sys
import math
import pickle
import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch



parser = argparse.ArgumentParser()
parser.add_argument("--st1", default=1, type=int, help="the start range of the first sudoku")
parser.add_argument("--ed1", default=4, type=int, help="the end range of the first sudoku")
parser.add_argument("--st2", default=4, type=int,help="the start range of the second sudoku")
parser.add_argument("--ed2", default=7, type=int,help="the end range of the second sudoku")
args = parser.parse_args()

assert (args.ed1 - args.st1 == 3)
assert (args.ed2 - args.st2 == 3)

base1 = 1
base2 = 5
shift1 = args.st1 - base1
shift2 = args.st2 - base2

img_size = 32
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1,
    shuffle=False,
)

dataloader1 = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1,
    shuffle=False,
)

num2img = {}
num2img28 = {}

for i, (img, label) in enumerate(dataloader):
    img = img[0].data.numpy()
    label = label[0].item()
    if label in num2img:
        num2img[label].append(img)
    else:
        num2img[label] = [img]

for i, (img, label) in enumerate(dataloader1):
    img = img[0].data.numpy()
    label = label[0].item()
    if label in num2img28:
        num2img28[label].append(img)
    else:
        num2img28[label] = [img]


sudoku = np.load("./4by4.npy") 
sudoku5678 = np.load("./4by4_5678.npy") 
ori_label = np.load("./4by4_labels.npy") #
ori_label5678 = np.load("./4by4_5678_labels.npy") 

sudoku_28 = np.zeros((sudoku.shape[0],sudoku.shape[1], sudoku.shape[2], 28, 28))
sudoku5678_28 = np.zeros((sudoku.shape[0],sudoku.shape[1], sudoku.shape[2], 28, 28))


print (sudoku.shape)
print (ori_label.shape)

for i in range(sudoku.shape[0]):
    for j in range(16):
        ori_label[i][j] += shift1
        ori_label5678[i][j] += shift2
        idx1 = np.random.choice(len(num2img[ori_label[i][j]]))
        idx2 = np.random.choice(len(num2img[ori_label5678[i][j]]))
        sudoku[i][j] = num2img[ori_label[i][j]][idx1]
        sudoku_28[i][j] = num2img28[ori_label[i][j]][idx1]
        sudoku5678[i][j] = num2img[ori_label5678[i][j]][idx2]
        sudoku5678_28[i][j] = num2img28[ori_label5678[i][j]][idx2]

print (sudoku.shape)
print (ori_label.shape)

sf1 = "./sudoku_%d_%d.npy" % (args.st1, args.ed1) 
if args.st2 == 1 and args.ed2 == 4:
    sf2 = "./sudoku_%d_%d_2.npy" % (args.st2, args.ed2) 
else:
    sf2 = "./sudoku_%d_%d.npy" % (args.st2, args.ed2) 
sf1_28 = "./sudoku_%d_%d_28.npy" % (args.st1, args.ed1) 
if args.st2 == 1 and args.ed2 == 4:
    sf2_28 = "./sudoku_%d_%d_28_2.npy" % (args.st2, args.ed2) 
else:
    sf2_28 = "./sudoku_%d_%d_28.npy" % (args.st2, args.ed2) 
lf1 = "./label_%d_%d.npy" % (args.st1, args.ed1) 
if args.st2 == 1 and args.ed2 == 4:
    lf2 = "./label_%d_%d_2.npy" % (args.st2, args.ed2) 
else:
    lf2 = "./label_%d_%d.npy" % (args.st2, args.ed2) 

np.save(sf1, sudoku)
np.save(sf2, sudoku5678)
np.save(sf1_28, sudoku_28)
np.save(sf2_28, sudoku5678_28)
np.save(lf1, ori_label)
np.save(lf2, ori_label5678)

      


