import os
import copy
import sys
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

def rescale(x):
    x = np.asarray(x)
    x = x.reshape(81, 1, 32, 32)
    #x = x.reshape(16, 1, 32, 32)
    return x

def convert(n, sudokus, digit_map, name, rate):
    ret = []
    ret_labels = []
    n_iter = n // len(sudokus) + 1
    offset = 1
    if "digit" in name:
        offset = 0
    for i in range(10):
        interval = int(len(digit_map[i]) // 4)
        np.random.seed(123)
        idx = np.arange(len(digit_map[i]))
        np.random.shuffle(idx)
        digit_map[i] = np.array(digit_map[i])
        digit_map[i] = digit_map[i][idx]
        digit_map[i] = digit_map[i][interval * rate:min(interval * (rate + 1), len(digit_map[i]))]
    for i in range(n_iter):
        for sudoku in sudokus:
            flatten = [number - offset for sublist in sudoku for number in sublist]
            ret_labels.append(flatten)
            emnist_sudoku = []
            for number in flatten:
                rnd = np.random.randint(len(digit_map[number]))
                emnist_sudoku.append(digit_map[number][rnd])
            ret.append(rescale(emnist_sudoku))
    return ret, ret_labels


seed = int(sys.argv[2])
np.random.seed(seed)

n = int(sys.argv[1])
img_sz = 32

#dataloader = torch.utils.data.DataLoader(
#    datasets.EMNIST(
#        "./emnist",
#        split="byclass",
#        train=True,
#        download=True,
#        transform=transforms.Compose(
#            [transforms.Resize(img_sz), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#        ),
#    ),
#    batch_size=1,
#    shuffle=True,
    #collate_fn=my_collate,
#)

sudokus = np.load("minimum.npy")[:, 1, :, :]
#sudokus = np.load("all4sudoku.npy")
label2digits = {}
label2upper = {}
#label2lower = {}
'''
tot = 0
for img, label in dataloader:
    tot += 1
    img = img.transpose(2, 3)
    label = label[0].item()
    img = img[0].data.numpy()
    if label >= 10:
        if label < 10 + 26:
            label = label - 10
            if label in label2upper:
                label2upper[label].append(img)
            else:
                label2upper[label] = [img]
        else:
            label = label - 36
            if label in label2lower:
                label2lower[label].append(img)
            else:
                label2lower[label] = [img]
    else:
        if label in label2digits:
            label2digits[label].append(img)
        else:
            label2digits[label] = [img]

'''
#label2digits = np.load("/atlaslocaldisk/shared/9by9_dichen/selected_digits.npy").item()
#label2upper = np.load("/atlaslocaldisk/shared/9by9_dichen/selected_uppers.npy").item()
label2digits = np.load("selected_digits_offset2.npy", allow_pickle=True).item()
label2upper = np.load("selected_uppers_offset2.npy", allow_pickle=True).item()

aaa = [label2digits, label2upper]
names = ["digit", "upper"]

rate_suffix = [(0, "train"), (1, "valid"), (2, "test")]
for item, name in zip(aaa, names):
    ori_digit_map = copy.deepcopy(item)
    for rate, suffix in rate_suffix:
        rtn, rtn_labels = convert(n, sudokus, item, name, rate)
        rtn, rtn_labels = rtn[:n], rtn_labels[:n]
        rtn, rtn_labels = np.array(rtn), np.array(rtn_labels)

        print (rtn.shape)
        print (rtn_labels.shape)

        s_rtn = []
        s_rtn_labels = []
        idx = np.arange(len(rtn))
        np.random.shuffle(idx)
        for i in idx:
            s_rtn.append(rtn[i])
            s_rtn_labels.append(rtn_labels[i])

        np.save("9by9_%s_%s.npy" % (name, suffix), s_rtn)
        np.save("9by9_%s_labels_%s.npy" % (name, suffix), s_rtn_labels)
        item = copy.deepcopy(ori_digit_map)






