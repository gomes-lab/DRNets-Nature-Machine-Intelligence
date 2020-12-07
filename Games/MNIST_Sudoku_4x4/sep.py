#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os, sys 
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch

from resnet import *

batch_size = 64
latent_dim = 100
n_classes = 10
img_shape = (1, 32, 32)
img_size = 32


# In[2]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #print ("label: " labels)
        #print ("!!!!!!!!!!!!!!!1 ", self.label_emb(labels).size(), " ", noise.size())
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


# In[3]:


generator = Generator()
if torch.cuda.is_available():
    print ("use cuda")
    generator = generator.cuda()
generator.load_state_dict(torch.load("models/G-180.model"))
generator.eval()


# In[4]:


dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)


# In[22]:


# Unsupervised sperate the image with dropout
from skimage import data
from skimage.transform import resize


nums = 4

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        #self.conv1 = torch.nn.Conv2d(1, 16, 3)
        #self.conv2 = torch.nn.Conv2d(16, 8, 3)
        
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, nums)
        self.fc4 = torch.nn.Linear(512, nums)
        self.softmax = torch.nn.Softmax(dim = 1)
    
    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #logit1 = self.fc3(x)
        #logit2 = self.fc4(x)
        #x1 = self.softmax((logit1.permute(1,0) - torch.max(logit1, dim = 1)[0]).permute(1, 0) ) #* ind1
        #x2 = self.softmax((logit2.permute(1,0) - torch.max(logit2, dim = 1)[0]).permute(1, 0) ) #* ind1
        x1 = self.softmax(self.fc3(x))
        x2 = self.softmax(self.fc4(x))
        return x1, x2

class seprator(nn.Module):
    def __init__(self):
        super(seprator, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 100 * nums * 2)
    def forward(self, x):
#         x = x.view(-1, 1024)
#         x = 3.0 * F.tanh(self.fc1(x))
#         x = 3.0 * F.tanh(self.fc2(x))
#         x = 3.0 * F.tanh(self.fc3(x)) - 0.5
#         x = x.view(1, 10, 100)
        
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = 3.0 * F.tanh(self.fc4(x))
        x = self.fc4(x)
        x = x.view(-1, 2, nums, 100)
        
        return x
    
def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim = 1), dim = 0)

def show_sudoku(x1, x2, x_mix, name="1"):
    n_col = 4
    n_row = 4
    fig, axes = plt.subplots(n_row, n_col * 3, figsize = (3 *n_col, n_row))
    for j in range(4):
        for k in range(4):
            axes[j][k].imshow(x1[j][k], cmap = "gray")
            axes[j][4 + k].imshow(x2[j][k], cmap="gray")
            axes[j][8 + k].imshow(x_mix[j][k], cmap="gray")
    plt.show()

def gen_alldiff_constraints(nums, batch_size):
    
    sqr_nums = int(np.sqrt(nums))
    idx = np.arange(nums**2).reshape(nums, nums)
    all_diffs = []
    for bs in range(batch_size):
        all_diff = []
        for i in range(nums):
            all_diff.append(idx[:,i])

        for i in range(nums):
            all_diff.append(idx[i,:])

        for i in range(sqr_nums):
            for j in range(sqr_nums):
                all_diff.append(idx[i*sqr_nums:(i+1)*sqr_nums, j*sqr_nums:(j+1)*sqr_nums].reshape(-1))
        all_diff = np.asarray(all_diff, dtype="int32")
        all_diff += bs * (nums**2)
        all_diffs.append(all_diff)

    all_diffs = np.concatenate(all_diffs, axis = 0)
    return all_diffs

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

parser = argparse.ArgumentParser()
parser.add_argument("--st1", default=1, type=int)
parser.add_argument("--ed1", default=4, type=int)
parser.add_argument("--st2", default=5, type=int)
parser.add_argument("--ed2", default=8, type=int)
parser.add_argument("--ori", default=0, type=int)
parser.add_argument("--ocase", default=0, type=int)
args = parser.parse_args()


pred_path = "./models/pred.model-9992"
sep_path = "./models/sep.model-9992"

#if args.ori:
#    sudoku = np.load("../../../4by4.npy") #100, 16, 1, 28, 28
#    sudoku5678 = np.load("../../../4by4_5678.npy") #100, 16, 1, 28, 28
#    ori_label = np.load("../../../4by4_labels.npy") #100, 16
#    ori_label5678 = np.load("../../../4by4_5678_labels.npy") #100, 16
#    base1 = 1
#    base2 = 5
#else:
#    sudoku = np.load("../../../sudoku_%d_%d.npy" % (args.st1, args.ed1)) #100, 16, 1, 28, 28
#    sudoku5678 = np.load("../../../sudoku_%d_%d.npy" % (args.st2, args.ed2)) #100, 16, 1, 28, 28
#    ori_label = np.load("../../../label_%d_%d.npy" % (args.st1, args.ed1)) #100, 16
#    ori_label5678 = np.load("../../../label_%d_%d.npy" % (args.st2, args.ed2)) #100, 16
#    base1 = args.st1
#    base2 = args.st2

ocase = args.ocase
print ("ocase: ", ocase)

if ocase == 1:
    sudoku = np.load("./sudoku_1_4.npy")
    sudoku5678 = np.load("./sudoku_4_7.npy")
    ori_label = np.load("./label_1_4.npy")
    ori_label5678 = np.load("./label_4_7.npy")
elif ocase == 2:
    sudoku = np.load("./sudoku_1_4.npy")
    sudoku5678 = np.load("./sudoku_3_6.npy")
    ori_label = np.load("./label_1_4.npy")
    ori_label5678 = np.load("./label_3_6.npy")
elif ocase == 3:
    sudoku = np.load("./sudoku_1_4.npy")
    sudoku5678 = np.load("./sudoku_2_5.npy")
    ori_label = np.load("./label_1_4.npy")
    ori_label5678 = np.load("./label_2_5.npy")
elif ocase == 4:
    sudoku = np.load("./sudoku_1_4.npy")
    sudoku5678 = np.load("./sudoku_1_4_2.npy")
    ori_label = np.load("./label_1_4.npy")
    ori_label5678 = np.load("./label_1_4_2.npy")
elif ocase == 0:
    sudoku = np.load("./4by4.npy") #100, 16, 1, 28, 28
    sudoku5678 = np.load("./4by4_5678.npy") #100, 16, 1, 28, 28
    ori_label = np.load("./4by4_labels.npy") #100, 16
    ori_label5678 = np.load("./4by4_5678_labels.npy") #100, 16


base1 = 1
base2 = 5 - ocase

n_data = sudoku.shape[0]

use_cuda = torch.cuda.is_available()

n_col = 4
n_row = 4

lr = 0.0001

sep = resnet18(predictor=False)
pred = resnet18(predictor=True)


if use_cuda:
    sep = sep.cuda()
    pred = pred.cuda()
    print("use_cuda")

def test_evaluate(pred, sep):
    pred = pred.eval()
    sep = sep.eval()
    s_mix_lst = []
    l1_lst = []
    l2_lst = []
    s1_lst = []
    s2_lst = []

    all_sudoku_acc = 0
    all_label_acc = 0
    all_recon_loss = 0
    cnt = 0

    for idx in range(10000):
        s1 = sudoku[idx]
        s2 = sudoku5678[idx]
        
        l1 = ori_label[idx]
        l2 = ori_label5678[idx]
        
        s1 = s1.reshape(nums, nums, 32, 32)
        s2 = s2.reshape(nums, nums, 32, 32)
        s_mix = np.maximum(s1, s2)
    
        s_mix = np.reshape(s_mix, (16, 1, 32, 32))
        
        #loading samples
        s_mix_lst.append(s_mix)
        l1_lst.append(l1)
        l2_lst.append(l2)
        s1_lst.append(s1)
        s2_lst.append(s2)
        
        if (len(s_mix_lst) == batch_size):
            s_mix = np.concatenate(s_mix_lst, axis = 0) # bs * 16, 1, 32, 32
            s_mix_copy = copy.deepcopy(s_mix)
            s_mix = Variable(torch.tensor(s_mix).float(), requires_grad=False)

            if use_cuda:
                s_mix = s_mix.cuda()
                
            #print ("Initial Image")
            #show_sudoku(s1_lst[0], s2_lst[0], np.reshape(s_mix_lst[0], (4, 4, 32, 32)), "s1")
            
            epochs = 1
            for ii in range(epochs):
                
                labels1_distribution, labels2_distribution = pred(s_mix) #bs * 16, 4 

                if (use_cuda):
                    z = sep(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float()).cuda() #bs * 16, 2, 4, 100
                else:
                    z = sep(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float())

                labels1 = labels1_distribution.cpu().data.numpy()
                labels2 = labels2_distribution.cpu().data.numpy()
                
                labels1_argmax = np.argmax(labels1, axis=1)
                labels2_argmax = np.argmax(labels2, axis=1)

                labels12 = np.concatenate([(labels1_argmax + base1).reshape(-1, 1), (labels2_argmax + base2).reshape(-1, 1)], axis = 1)
                
                l1 = np.concatenate(l1_lst, axis = 0)
                l2 = np.concatenate(l2_lst, axis = 0)
                
                l12 = np.concatenate([l1.reshape(-1, 1), l2.reshape(-1, 1)], axis = 1) # bs * 16, 2
                
                eqn = np.equal(labels12, l12).astype("int").reshape(batch_size, nums**2, 2)
                
                label_acc = np.mean((np.sum(eqn, axis = 2) == 2).astype("float32"))
                sudoku_acc = np.mean((np.sum(eqn, axis = (1,2)) == 32).astype("float32"))
                
                # compute mixture


                gen_imgs = generator(z.view(-1, 100), gen_labels) #bs*16*2*4, 1, 32, 32

                label_distribution = torch.cat([labels1_distribution, labels2_distribution], dim = 1) # bs * 16 * 8

                gen_mix = gen_imgs.permute(1, 2, 3, 0) * label_distribution.view(-1)

                gen_mix = gen_mix.view(1, 32, 32, batch_size * 16, 2, 4)

                gen_mix = torch.sum(gen_mix, dim = 5) # avg by distribution 1, 32, 32, bs*16, 2

                gen_img_demix = gen_mix.permute(3, 4, 0, 1, 2) # bs*16, 2, 32, 32 #only used for visualization

                #gen_mix = torch.max(gen_mix, dim = 4)[0]
                gen_mix = torch.mean(gen_mix, dim=4)

                gen_mix = gen_mix.permute(3, 0, 1, 2).view(-1, 32, 32) #bs * 16, 32, 32

                all_label_acc += label_acc
                all_sudoku_acc += sudoku_acc
                cnt += 1
    return all_label_acc / cnt, all_sudoku_acc / cnt

print ("n_data: ", n_data)
print ("base1: ", base1)
print ("base2: ", base2)

#n_data = 1000
batch_size = 100
check_freq = 100

s_mix_lst = []
l1_lst = []
l2_lst = []
s1_lst = []
s2_lst = []

alldiff_constraints = gen_alldiff_constraints(nums, batch_size) #bs * 12 * 4

if args.ori:
    gen_labels = torch.LongTensor(np.arange(nums**3 * 2 * batch_size, dtype = "int32") % (2*nums) + base1)
else:
    labels = []
    for i in range(nums ** 2 * batch_size):
        for j in range(args.st1, args.ed1 + 1):
            labels.append(j)
        for j in range(args.st2, args.ed2 + 1):
            labels.append(j)
    gen_labels = torch.LongTensor(labels)

if use_cuda:
    gen_labels =  gen_labels.cuda()
    
print("Training Starts")

for _epoch_ in range(10000):
    all_sudoku_acc = 0
    all_label_acc = 0
    all_recon_loss = 0
    cnt = 0
    
    
    for idx in range(n_data):
    
        s1 = sudoku[idx]
        s2 = sudoku5678[idx]
        
        l1 = ori_label[idx]
        l2 = ori_label5678[idx]
        
        s1 = s1.reshape(nums, nums, 32, 32)
        s2 = s2.reshape(nums, nums, 32, 32)
        s_mix = np.maximum(s1, s2)
    
        s_mix = np.reshape(s_mix, (16, 1, 32, 32))
        
        #loading samples
        s_mix_lst.append(s_mix)
        l1_lst.append(l1)
        l2_lst.append(l2)
        s1_lst.append(s1)
        s2_lst.append(s2)
        
        if (len(s_mix_lst) == batch_size):
            s_mix = np.concatenate(s_mix_lst, axis = 0) # bs * 16, 1, 32, 32
            s_mix = Variable(torch.tensor(s_mix).float(), requires_grad=False)

            if use_cuda:
                s_mix = s_mix.cuda()
                
            #print ("Initial Image")
            #show_sudoku(s1_lst[0], s2_lst[0], np.reshape(s_mix_lst[0], (4, 4, 32, 32)), "s1")
            
            epochs = 1
            for ii in range(epochs):
                
                labels1_distribution, labels2_distribution = pred(s_mix) #bs * 16, 4 

                if (use_cuda):
                    z = sep(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float()).cuda() #bs * 16, 2, 4, 100
                else:
                    z = sep(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float())

                optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()), lr=lr)
                #optimizer = torch.optim.Adam(list(sep.parameters()), lr=lr)
                
                # compute accs
                labels1 = labels1_distribution.cpu().data.numpy()
                labels2 = labels2_distribution.cpu().data.numpy()
                
                labels1_argmax = np.argmax(labels1, axis=1)
                labels2_argmax = np.argmax(labels2, axis=1)

                labels12 = np.concatenate([(labels1_argmax + base1).reshape(-1, 1), (labels2_argmax + base2).reshape(-1, 1)], axis = 1)
                
                l1 = np.concatenate(l1_lst, axis = 0)
                l2 = np.concatenate(l2_lst, axis = 0)
                
                l12 = np.concatenate([l1.reshape(-1, 1), l2.reshape(-1, 1)], axis = 1) # bs * 16, 2
                
                
                eqn = np.equal(labels12, l12).astype("int").reshape(batch_size, nums**2, 2)
                
                label_acc = np.mean((np.sum(eqn, axis = 2) == 2).astype("float32"))
                sudoku_acc = np.mean((np.sum(eqn, axis = (1,2)) == 32).astype("float32"))
                
                # compute mixture


                gen_imgs = generator(z.view(-1, 100), gen_labels) #bs*16*2*4, 1, 32, 32

                label_distribution = torch.cat([labels1_distribution, labels2_distribution], dim = 1) # bs * 16 * 8

                gen_mix = gen_imgs.permute(1, 2, 3, 0) * label_distribution.view(-1)

                gen_mix = gen_mix.view(1, 32, 32, batch_size * 16, 2, 4)

                gen_mix = torch.sum(gen_mix, dim = 5) # avg by distribution 1, 32, 32, bs*16, 2

                gen_img_demix = gen_mix.permute(3, 4, 0, 1, 2) # bs*16, 2, 32, 32 #only used for visualization

                gen_mix = torch.max(gen_mix, dim = 4)[0]

                gen_mix = gen_mix.permute(3, 0, 1, 2).view(-1, 32, 32) #bs * 16, 32, 32

                cri = torch.nn.L1Loss()

                loss_recon = 0.

                loss_recon = cri(s_mix.view(-1, 32, 32), gen_mix)

                loss_recon /= (1.0 * labels1_distribution.size(0))

                entropy_cell = 0.5 * (entropy(labels1_distribution) + entropy(labels2_distribution))
                
                all_diff_loss1 = entropy(torch.mean(labels1_distribution[torch.LongTensor(alldiff_constraints)], dim = 1))
                all_diff_loss2 = entropy(torch.mean(labels2_distribution[torch.LongTensor(alldiff_constraints)], dim = 1))
                entropy_alldiff = 0.5 * (all_diff_loss1 + all_diff_loss2) 

                scale_recon = 0.001

                keep_p = 1.0

                drop_out_recon = torch.nn.Dropout(p = 1.0 - keep_p)
                drop_out_cell = torch.nn.Dropout(p = 1.0 - 1.0)
                drop_out_alldiff = torch.nn.Dropout(p = 1.0 - 1.0)


                loss_recon = drop_out_recon(loss_recon)
                entropy_cell_drop = drop_out_cell(entropy_cell)
                entropy_alldiff_drop = drop_out_alldiff(entropy_alldiff)

                if (_epoch_ < 0):
                    loss = scale_recon * loss_recon
                else:
                    #loss = scale_recon * loss_recon
                    loss = scale_recon * loss_recon +  0.01 * entropy_cell_drop - 1.0 * (entropy_alldiff_drop)
            
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if (ii % 100 == 0 and (cnt + 1) % check_freq == 0):
                    #print ("Initial Image")
                    #show_sudoku(s1_lst[0], s2_lst[0], np.reshape(s_mix_lst[0], (4, 4, 32, 32)), "s1")
                    print ("Iteration %d: total_loss: %f, loss_recon: %f, entropy_cell: %f, entropy_alldiff: %f"                     % (ii, loss.item(), scale_recon * loss_recon.item(), entropy_cell.item(),  entropy_alldiff.item()))
                    print ("epoch = %d, iter-%d, label_acc = %f, sudoku_acc= %f" % (_epoch_, ii, label_acc, sudoku_acc))

                    for i in range(4):
                        for j in range(4):
                            print(labels1_argmax[i*4 + j] + base1, end = ",")
                        print(" ", end = "")

                        for j in range(4):
                            print(labels2_argmax[i*4 + j] + base2, end = ",")
                        print(" ", end = "")

                        for j in range(4):
                            print(l1[i*4 + j], end = ",")
                        print(" ", end = "")

                        for j in range(4):
                            print(l2[i*4 + j], end = ",")
                        print("")

                    
                    #gen_imgs1_numpy = np.concatenate([item.cpu().data.numpy() for item in gen_img_demix[:nums**2,0]])
                    #gen_imgs1_numpy = np.reshape(gen_imgs1_numpy, (4, 4, 32, 32))

                    #gen_imgs2_numpy = np.concatenate([item.cpu().data.numpy() for item in gen_img_demix[:nums**2,1]])
                    #gen_imgs2_numpy = np.reshape(gen_imgs2_numpy, (4, 4, 32, 32))

                    #gen_mix_numpy = np.concatenate([item.cpu().data.numpy() for item in gen_mix[:nums**2]])
                    #gen_mix_numpy = np.reshape(gen_mix_numpy, (4, 4, 32, 32))
                    
                    #show_sudoku(gen_imgs1_numpy, gen_imgs2_numpy, gen_mix_numpy)
            
            s_mix_lst = []
            l1_lst = []
            l2_lst = []
            s1_lst = []
            s2_lst = []
                 
            all_label_acc += label_acc
            all_sudoku_acc += sudoku_acc
            all_recon_loss += scale_recon * loss_recon.item()
            cnt += 1
        
            if (cnt % check_freq == 0):

                #print ("Initial Image")
                #show_sudoku(s1, s2, np.reshape(s_mix, (4, 4, 32, 32)), "s1")

                print("#puzzle = %d, sudoku_acc = %f, label_acc = %f, recon_loss = %f"%(cnt * batch_size, all_sudoku_acc/cnt, all_label_acc/cnt, all_recon_loss/cnt))
                save_model(pred, pred_path)
                save_model(sep, sep_path)

                cnt = 0
                all_label_acc = 0
                all_sudoku_acc = 0
                all_recon_loss = 0

                #test_label_acc, test_sudoku_acc = test_evaluate(pred, sep)
                #print ("!!!test results, sudoku_acc: %f, label_acc: %f" % (test_sudoku_acc, test_label_acc))
                sys.stdout.flush()
                #pred = pred.train()
                #sep  = sep.train()

         
