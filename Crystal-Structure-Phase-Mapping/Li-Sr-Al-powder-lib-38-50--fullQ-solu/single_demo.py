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

    raw_xrd = np.load(FLAGS.edges_dir[:-9] + "raw_XRD.npy")
    Q_idx = np.load(FLAGS.Q_idx_dir)
    Q = np.load(FLAGS.Q_dir)
    idx = int(sys.argv[1]) - 1
    ##############################
    f, axes = plt.subplots(2, 1, figsize = (10, 2.5))
    x = xrd[idx][Q_idx]
    axes[0].plot(Q, rescale(x), color = "b")
    axes[0].legend(["xrd_std of #%d, composition = %s"%(idx, composition[idx])], loc = "upper right")

    axes[1].plot(rescale(raw_xrd[idx]), color = "b")
    plt.show()
    ######################################################
    edges = np.load(FLAGS.edges_dir)
    for i in range(edges.shape[1]):
        if (edges[idx][i] == 1):
            print(i)
    
if __name__=='__main__':
    tf.app.run()



