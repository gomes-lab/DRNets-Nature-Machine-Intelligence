import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import model
import get_data 
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import urllib
import os, sys
#from pyheatmap.heatmap import HeatMap
#import seaborn as sns

FLAGS = tf.app.flags.FLAGS
def stats(name, x, visual = True):
    idx = np.argsort(x)
    print("stats of %s min:"%name, np.min(x), "max:", np.max(x), "mean", np.mean(x), "median", np.median(x), "std", np.std(x), "worse_idx", idx[-1])
    if (visual):
        plt.hist(x, 100)
        plt.title(name)
        plt.show()
    return idx

def rescale(x):
    return x / (FLAGS.eps + np.max(x))
    #* FLAGS.peak_rescale  #/ (1e-9 + np.max(x))

def compute_XRD(mu, intensity):
    Q = np.load(FLAGS.Q_dir)
    y = np.zeros_like(Q)
    x = Q 
    max_I = np.max(intensity)
    for i in range(FLAGS.n_spike):
        pos = int((mu[i] - Q[0])/(Q[-1] - Q[0])*Q.shape[0])
        if (pos >= 0 and pos < FLAGS.xrd_dim):
            y[pos] += intensity[i]/max_I

    return y

def visual_bases_sol(bases_sol):
    M = bases_sol.shape[0]
    #print("M=", M)
    f, axes = plt.subplots(M, 1, figsize = (15, M*2.5))
    for i in range(M):
        ax = axes[i]
        x = bases_sol[i]
        x = rescale(x)
        ax.plot(x)
    plt.show()
    #plt.savefig("bases.png")

def comp2Coords(comp):

    vec2 = np.array([0,0])
    vec3 = np.array([1,0])
    vec1 = np.array([1.0/2, np.sqrt(3.0/4)])
    array = []


    for i in range(len(comp)):
        array.append(comp[i][0]*vec1 + comp[i][1]*vec2 + comp[i][2]*vec3)

    return np.asarray(array) # (x_i, y_i)


def main(_):

    print('reading npy...')
    #np.random.seed(19950420) # set the random seed of numpy 
    data, batches = get_data.get_data() #XRD sources and batches
    train_idx = np.arange(batches.shape[0]) #load the indices of the training set

    xrd = get_data.get_xrd_mat()
    composition = get_data.get_comp()
    degree_of_freedom = get_data.get_degree_of_freedom(np.arange(FLAGS.testing_size))

    Q_idx = np.load(FLAGS.Q_idx_dir)
    Q = np.load(FLAGS.Q_dir)
    minors = []
    ######################################################
    for i in range(xrd.shape[0]):
        x = rescale(xrd[i])
        p = x / np.sum(x)
        minor = 0
        for j in range(xrd.shape[1]):
            if (x[j] <= 0.05):
                minor += p[j]
        minors.append(minor)

    stats("minor", minors)


    ##############################
    indices = np.arange(xrd.shape[0])

    if ("binary" in sys.argv):
        tmp = []
        for i in indices:
            if (degree_of_freedom[i] == 2.0):
                tmp.append(i)
        indices = tmp 

    N = len(indices)
    i = 0

    if ("start" in sys.argv):
        i = int(sys.argv[-1])
    print(N)
    while (i < N):
        #plt.clf()
        f, axes = plt.subplots(6, 2, figsize = (10*2, 6*2.5))
        #print(axes)
        #dcp = (decomposition_sol[i].T / (1e-9 + np.max(decomposition_sol[i], axis = 1)) )
        #xrd_recon = np.sum(decomposition_sol[i], axis = 0) #np.dot(dcp, weights_sol[i])
        #xrd_std = xrd[i] #/ np.max(xrd[i])
        #print(np.sum(np.abs(xrd_std - xrd_recon)))
        for j in range(6):
            for k in range(2):
                if (i + k * 6 + j < N):
                    idx = indices[i + k * 6 + j]
                    axes[j][k].plot(Q, rescale(xrd[idx][Q_idx]), color = "b")
                    axes[j][k].legend(["xrd_std of #%d, composition = %s"%(idx, composition[idx])], loc = "upper right")

        i += 12
        plt.show()
    ######################################################

    
if __name__=='__main__':
    tf.app.run()



