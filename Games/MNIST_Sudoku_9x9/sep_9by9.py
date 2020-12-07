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

from resnet_9by9 import *

np.random.seed(19950419)
torch.manual_seed(19950419)
torch.cuda.manual_seed_all(19950419)

batch_size = 64
latent_dim = 100
n_classes = 10
img_shape = (1, 32, 32)
img_size = 32


# In[2]:
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GREY = '\033[90m'

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


generator_digit = Generator()
generator_upper = Generator()
if torch.cuda.is_available():
    print ("use cuda")
    generator_digit = generator_digit.cuda()
    generator_upper = generator_upper.cuda()
generator_digit.load_state_dict(torch.load("./models/G-180.model"))
generator_upper.load_state_dict(torch.load("./models/upper_emnist_G-180.model"))
generator_digit.eval()
generator_upper.eval()


from skimage import data
from skimage.transform import resize


nums = 9

def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim = 1), dim = 0)

def show_sudoku(x1, x2, x_mix, name="1"):
    n_col = 9 
    n_row = 9
    fig, axes = plt.subplots(n_row, n_col * 3, figsize = (3 *n_col, n_row))
    for j in range(n_row):
        for k in range(n_col):
            axes[j][k].imshow(x1[j][k], cmap = "gray")
            axes[j][9 + k].imshow(x2[j][k], cmap="gray")
            axes[j][18 + k].imshow(x_mix[j][k], cmap="gray")
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

def load_model(model, path, is_eval=True):
    model.load_state_dict(torch.load(path))
    if (is_eval):
        model.eval()

def save_data(x, name):
    print(name, x.shape)
    np.save(name, x)

parser = argparse.ArgumentParser()
parser.add_argument("--st1", default=1, type=int)
parser.add_argument("--ed1", default=4, type=int)
parser.add_argument("--st2", default=5, type=int)
parser.add_argument("--ed2", default=8, type=int)
parser.add_argument("--ori", default=0, type=int)
parser.add_argument("--ocase", default=0, type=int)
parser.add_argument("--batch_size", default=40, type=int)
parser.add_argument("--clip_step", default=10000, type=int)
parser.add_argument("--epoch", default=1000, type=int)
parser.add_argument("--all_diff", default=1.0, type=float)
parser.add_argument("--entropy_cell", default=0.01, type=float)
parser.add_argument("--scale_recon", default=0.001, type=float)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--model_idx", default="now2", type=str)
parser.add_argument("--restart", default=False, type=bool)

args = parser.parse_args()

print(args)

pred_path = "./models/pred.model-%s" % args.model_idx
sep_path = "./models/sep.model-%s" % args.model_idx
sep2_path = "./models/sep2.model-%s" % args.model_idx 

ocase = args.ocase
print ("ocase: ", ocase)

if ocase == 1:
    pass
elif ocase == 2:
    pass
elif ocase == 3:
    pass
elif ocase == 4:
    pass
elif ocase == 0:
    sudoku = np.load("./9by9_digit_test.npy") #100, 16, 1, 28, 28
    sudoku5678 = np.load("./9by9_upper_test.npy") #100, 16, 1, 28, 28
    ori_label = np.load("./9by9_digit_labels_test.npy") #100, 16
    ori_label5678 = np.load("./9by9_upper_labels_test.npy") #100, 16

sudoku = sudoku[:args.clip_step]
sudoku5678 = sudoku5678[:args.clip_step]
ori_label = ori_label[:args.clip_step]
ori_label5678 = ori_label5678[:args.clip_step]

print(sudoku.shape)

base1 = 1
base2 = 1
#base2 = 5 - ocase

n_data = sudoku.shape[0]

use_cuda = torch.cuda.is_available()

n_col = 9
n_row = 9

lr = 0.0001

N_epoch = args.epoch

sep = resnet18(predictor=False)
sep2 = resnet18(predictor=False)
pred = resnet18(predictor=True)

if (args.mode == "test"):
    print("loading trained models", pred_path)
    load_model(pred, pred_path)
    load_model(sep, sep_path)
    load_model(sep2, sep2_path)
    N_epoch = 1

if (args.mode == "con"):
    print("loading trained models", pred_path)
    load_model(pred, pred_path, False)
    load_model(sep, sep_path, False)
    load_model(sep2, sep2_path, False)

if use_cuda:
    sep = sep.cuda()
    sep2 = sep2.cuda()
    pred = pred.cuda()
    print("use_cuda")

print ("n_data: ", n_data)
print ("base1: ", base1)
print ("base2: ", base2)

#n_data = 1000
batch_size = args.batch_size
check_freq = 1 #int(args.clip_step // batch_size)

s_mix_lst = []
l1_lst = []
l2_lst = []
s1_lst = []
s2_lst = []

alldiff_constraints = gen_alldiff_constraints(nums, batch_size) #bs * 12 * 4

if args.ori:
    pass
else:
    labels1 = []
    labels2 = []
    for i in range(nums ** 2 * batch_size):
        for j in range(1, 10):
            labels1.append(j)
        for j in range(9):
            labels2.append(j)
    gen_labels1 = torch.LongTensor(labels1)
    gen_labels2 = torch.LongTensor(labels2)

if use_cuda:
    gen_labels1 =  gen_labels1.cuda()
    gen_labels2 =  gen_labels2.cuda()
    
print("Training Starts")
best_acc = 0
entropy_factor = 1

indices = np.arange(n_data, dtype = "int32")
if (args.restart):
    indices = np.load("unsolved.npy").astype("int")
    indices = list(indices)
    for i in range(100 - len(indices)):
        indices.append(np.random.randint(n_data))
    print("Restart mode")
    print("unsolved:", len(indices))

for _epoch_ in range(N_epoch):
    all_sudoku_acc = 0
    all_label_acc = 0
    all_recon_loss = 0
    cnt = 0
    
    pred_probs1 = []
    pred_probs2 = []
    gt_labels1 = []
    gt_labels2 = []
    
    np.random.shuffle(indices)
    
    batches = []
    for idx in indices:
        s1 = sudoku[idx]
        s2 = sudoku5678[idx]
        #save_image(torch.tensor(s1).float(), "images/target1.png", nrow=9)
        #save_image(torch.tensor(s2).float(), "images/target2.png", nrow=9)
        
        l1 = ori_label[idx]
        l2 = ori_label5678[idx]
        
        s1 = s1.reshape(nums, nums, 32, 32)
        s2 = s2.reshape(nums, nums, 32, 32)
        s_mix = np.maximum(s1, s2)
    
        s_mix = np.reshape(s_mix, (nums ** 2, 1, 32, 32))
        
        #loading samples
        s_mix_lst.append(s_mix)
        l1_lst.append(l1)
        l2_lst.append(l2)
        s1_lst.append(s1)
        s2_lst.append(s2)
        
        if (len(s_mix_lst) == batch_size):
            s_mix = np.concatenate(s_mix_lst, axis = 0) # bs * 16, 1, 32, 32
            s_mix = Variable(torch.tensor(s_mix).float(), requires_grad=False)
            save_image(s_mix, "images/target.png", nrow=9)

            if use_cuda:
                s_mix = s_mix.cuda()
           
            #print ("Initial Image")
            #show_sudoku(s1_lst[0], s2_lst[0], np.reshape(s_mix_lst[0], (4, 4, 32, 32)), "s1")
            
            epochs = 1
            for ii in range(epochs):

                batches.append(s_mix.cpu())
                labels1_distribution, labels2_distribution = pred(s_mix) #bs * 16, 4 

                if (use_cuda):
                    z1 = sep(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float()).cuda() #bs * 16, 2, 4, 100
                    z2 = sep2(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float()).cuda() #bs * 16, 2, 4, 100
                else:
                    z1 = sep(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float())
                    z2 = sep2(torch.tensor(s_mix.reshape(-1, 1, 32, 32)).float())

                optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()) + list(sep2.parameters()), lr=lr)
                #optimizer = torch.optim.Adam(list(sep.parameters()), lr=lr)
                
                # compute accs
                labels1 = labels1_distribution.cpu().data.numpy()
                labels2 = labels2_distribution.cpu().data.numpy()
                
                #pred_probs1.append(labels1)
                #pred_probs2.append(labels2)

                labels1_argmax = np.argmax(labels1, axis=1)
                labels2_argmax = np.argmax(labels2, axis=1)

                labels12 = np.concatenate([(labels1_argmax + base1).reshape(-1, 1), (labels2_argmax).reshape(-1, 1)], axis = 1)
                
                l1 = np.concatenate(l1_lst, axis = 0)
                l2 = np.concatenate(l2_lst, axis = 0)
                gt_labels1.append(l1)
                gt_labels2.append(l2)
                
                l12 = np.concatenate([l1.reshape(-1, 1), l2.reshape(-1, 1)], axis = 1) # bs * 16, 2
                
                
                eqn = np.equal(labels12, l12).astype("int").reshape(batch_size, nums**2, 2)
                
                label_acc = np.mean((np.sum(eqn, axis = 2) == 2).astype("float32"))
                sudoku_acc = np.mean((np.sum(eqn, axis = (1,2)) == 2 * (nums ** 2)).astype("float32"))
                
                # compute mixture

                gen_img1 = generator_digit(z1.view(-1, latent_dim), gen_labels1) #bs*16*2*4, 1, 32, 32
                gen_img2 = generator_upper(z2.view(-1, latent_dim), gen_labels2) #bs*16*2*4, 1, 32, 32
                #print (gen_img1.size())
                
                gen_img1 = gen_img1.view((nums ** 2) * batch_size, nums, 1, 32, 32)
                gen_img2 = gen_img2.view((nums ** 2) * batch_size, nums, 1, 32, 32)
                gen_imgs = torch.cat([gen_img1, gen_img2], dim=1)
                #print (gen_imgs.size())
                gen_imgs = gen_imgs.view(-1, 1, 32, 32)
                #print (gen_imgs.size())
                #gen_imgs = torch.cat([gen_img1, gen_img2], dim=0)


                label_distribution = torch.cat([labels1_distribution, labels2_distribution], dim = 1) # bs * 16 * 8

                gen_mix = gen_imgs.permute(1, 2, 3, 0) * label_distribution.view(-1)

                gen_mix = gen_mix.view(1, 32, 32, batch_size * nums * nums, 2, nums)

                gen_mix = torch.sum(gen_mix, dim = 5) # avg by distribution 1, 32, 32, bs*16, 2

                gen_img_demix = gen_mix.permute(3, 4, 0, 1, 2) # bs*16, 2, 32, 32 #only used for visualization
                #save_image(gen_img_demix[:, 0, :, :, :], "images/sep1_%d.png" % (_epoch_), nrow=9)
                #save_image(gen_img_demix[:, 1, :, :, :], "images/sep2_%d.png" % (_epoch_), nrow=9)

                gen_mix = torch.max(gen_mix, dim = 4)[0]

                gen_mix = gen_mix.permute(3, 0, 1, 2).view(-1, 32, 32) #bs * 16, 32, 32
                
                #gen_mix_extend = gen_mix[:, None, :, :]
                #save_image(gen_mix_extend, "images/mix%d.png" % (_epoch_), nrow=9)

                cri = torch.nn.L1Loss()

                loss_recon = 0.

                loss_recon = cri(s_mix.view(-1, 32, 32), gen_mix)

                loss_recon /= (1.0 * labels1_distribution.size(0))

                entropy_cell = 0.5 * (entropy(labels1_distribution) + entropy(labels2_distribution))
                
                all_diff_loss1 = entropy(torch.mean(labels1_distribution[torch.LongTensor(alldiff_constraints)], dim = 1))
                all_diff_loss2 = entropy(torch.mean(labels2_distribution[torch.LongTensor(alldiff_constraints)], dim = 1))
                entropy_alldiff = 0.5 * (all_diff_loss1 + all_diff_loss2) 

                scale_recon = args.scale_recon
               # 0.005

                keep_p = 1.0

                drop_out_recon = torch.nn.Dropout(p = 1.0 - keep_p)
                drop_out_cell = torch.nn.Dropout(p = 1.0 - 1.0)
                drop_out_alldiff = torch.nn.Dropout(p = 1.0 - 1.0)


                loss_recon = drop_out_recon(loss_recon)
                entropy_cell_drop = drop_out_cell(entropy_cell)
                entropy_alldiff_drop = drop_out_alldiff(entropy_alldiff)
                
                if (_epoch_ < 5 and args.mode == "train"):
                    loss = args.scale_recon * loss_recon
                else:
                    loss = args.scale_recon * loss_recon +  args.entropy_cell * entropy_factor * entropy_cell_drop - args.all_diff * (entropy_alldiff_drop)
                
                if (args.mode != "test"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #if (ii % batch_size == 0 and (cnt + 1) % check_freq == 0):
                if (cnt % 10 == 0): 
                    #print ("Initial Image")
                    #show_sudoku(s1_lst[0], s2_lst[0], np.reshape(s_mix_lst[0], (4, 4, 32, 32)), "s1")
                    print ("Iteration %d: total_loss: %f, loss_recon: %f, entropy_cell: %f, entropy_alldiff: %f"\
                    % (cnt, loss.item(), loss_recon.item(), entropy_cell.item(),  entropy_alldiff.item()))
                    print ("epoch = %d, iter-%d, label_acc = %f, sudoku_acc= %f" % (_epoch_, ii, label_acc, sudoku_acc))

                    for i in range(nums):
                        for j in range(nums):
                            x = labels1_argmax[i*nums + j] + base1
                            gt_x = l1[i*nums + j]
                            if (x == gt_x):
                                print(x, end = ",")
                            else:
                                print(color.RED + "%d"%x + color.END, end = ",")
                        print(" ", end = "")

                        for j in range(nums):
                            x = labels2_argmax[i*nums + j] + base2
                            gt_x = l2[i*nums + j] + base2
                            if (x == gt_x):
                                print(x, end = ",")
                            else:
                                print(color.RED + "%d"%x + color.END, end = ",")
                        print(" ", end = "")

                        for j in range(nums):
                            print(l1[i*nums + j], end = ",")
                        print(" ", end = "")

                        for j in range(nums):
                            print(l2[i*nums + j] + base2, end = ",")
                        print("")

                    
            s_mix_lst = []
            l1_lst = []
            l2_lst = []
            s1_lst = []
            s2_lst = []
                 
            all_recon_loss += loss_recon.item()
            all_sudoku_acc += sudoku_acc
            all_sudoku_acc += sudoku_acc
            cnt += 1
        
    if (_epoch_ % check_freq == 0):
        print("validating performance")
        all_sudoku_acc = []
        all_label_acc = []
        unsolved = []
        for i in range(len(batches)):
            s_mix = batches[i]
            if (use_cuda):
                s_mix = s_mix.cuda()

            labels1_distribution, labels2_distribution = pred(s_mix) #bs * 16, 4 
            labels1 = labels1_distribution.cpu().data.numpy()
            labels2 = labels2_distribution.cpu().data.numpy()
             
            pred_probs1.append(labels1)
            pred_probs2.append(labels2)

            l1 = gt_labels1[i]
            l2 = gt_labels2[i]
            labels1_argmax = np.argmax(labels1, axis=1)
            labels2_argmax = np.argmax(labels2, axis=1)

            labels12 = np.concatenate([(labels1_argmax + base1).reshape(-1, 1), (labels2_argmax).reshape(-1, 1)], axis = 1)          
            l12 = np.concatenate([l1.reshape(-1, 1), l2.reshape(-1, 1)], axis = 1) # bs * 16, 2   
            eqn = np.equal(labels12, l12).astype("int").reshape(batch_size, nums**2, 2)
            label_acc = np.mean((np.sum(eqn, axis = 2) == 2).astype("float32"))
            sudoku_acc = (np.sum(eqn, axis = (1,2)) == 2 * (nums ** 2)).astype("float32")

            for j in range(len(sudoku_acc)):
                if (sudoku_acc[j] == 0):
                    unsolved.append(indices[i*batch_size + j])
            
            sudoku_acc = np.mean(sudoku_acc)
            all_sudoku_acc.append(sudoku_acc)
            all_label_acc.append(label_acc)
        
        all_sudoku_acc = np.mean(all_sudoku_acc)
        all_label_acc = np.mean(all_label_acc)
        if (len(unsolved) <= 10):
            print(unsolved)
        if (args.mode == "test"): 
            np.save("unsolved", unsolved)
        print("#puzzle = %d, sudoku_acc = %f, label_acc = %f, recon_loss = %f (best_acc = %f)"%\
        (cnt * batch_size, all_sudoku_acc, all_label_acc, all_recon_loss/cnt, best_acc))
                
        if (args.mode != "test" and all_sudoku_acc > best_acc):
            best_acc = all_sudoku_acc
            save_model(pred, pred_path)
            save_model(sep, sep_path)
            save_model(sep2, sep2_path) 
            if (best_acc > 0.97):
                entropy_factor = 1.0
                    
        cnt = 0
        all_label_acc = 0
        all_sudoku_acc = 0
        all_recon_loss = 0
        sys.stdout.flush()

pred_probs1 = np.concatenate(pred_probs1, axis = 0)
pred_probs2 = np.concatenate(pred_probs2, axis = 0)
gt_labels1 = np.concatenate(gt_labels1, axis = 0)
gt_labels2 = np.concatenate(gt_labels2, axis = 0)

save_data(pred_probs1, "DRNet-res/pred_probs1")
save_data(pred_probs2, "DRNet-res/pred_probs2")
save_data(gt_labels1, "DRNet-res/gt_labels1")
save_data(gt_labels2, "DRNet-res/gt_labels2")
