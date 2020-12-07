import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

FLAGS = tf.app.flags.FLAGS

data = np.load(FLAGS.data_dir)
batches = np.load(FLAGS.batches_dir)
Q = np.load(FLAGS.Q_dir)
Q_idx = np.load(FLAGS.Q_idx_dir)

sample_indicator = np.load(FLAGS.sample_indicator_dir)

degree_of_freedom = np.load(FLAGS.degree_of_freedom_dir)

refined_sample_indicator = None 
#np.load(FLAGS.refined_sample_indicator_dir)
refined_shift_indicator = None 
#np.load(FLAGS.refined_shift_indicator_dir)

_features = data
_mean = np.mean(_features, axis=0)
_std = np.std(_features, axis=0)
#print("means:", _mean)
#print("stds:", _std)

def get_data():
    return data, batches

def get_xrd_mat():
    return data[:,FLAGS.feature_dim:]

def get_comp():
    return data[:,:FLAGS.feature_dim]

def get_weights_sol():
    return np.load(FLAGS.weights_sol_dir)

def get_bases_sol():
    return np.load(FLAGS.bases_sol_dir)

def get_decomposition_sol():
    return np.load(FLAGS.decomposition_sol_dir)

def get_feature(data, my_order, offset = None):
    # we use (x - mean(x))/stderr as the normalization
    # we remove the lattice parameters and G6
    output = []
    if (offset == None):
        offset = 0
        
    for i in my_order:
        x = data[i][offset:offset + FLAGS.feature_dim]
        #x = (x - _mean)/_std
        output.append(x)

    output = np.array(output, dtype="float32") 
    return output

def get_xrd(data, my_order, offset = None):
    # we use max-normalization
    output = []
    if (offset == None):
        offset = FLAGS.feature_dim
        
    for i in my_order:
        x = data[i][offset:offset + FLAGS.xrd_dim]
        x = x[Q_idx]
        #x = x /(np.max(x) + FLAGS.eps)
        #x * FLAGS.peak_rescale #/(np.max(x) + 1e-9)
        output.append(x)

    output = np.array(output, dtype="float32") 
    return output

def get_indicator(my_order, flag = 0, offset = None):
    # we use (x - mean(x))/stderr as the normalization
    # we remove the lattice parameters and G6
    output = []
    rand_I = np.random.randint(2, size = FLAGS.n_bases + FLAGS.n_new_bases)
    for i in my_order:
        v = flag 
        x = np.concatenate([sample_indicator[i][:FLAGS.n_bases], np.zeros(FLAGS.n_new_bases) + v])

        if (FLAGS.rand_I == 1):
            x *= rand_I
        #x = (x - _mean)/_std
        output.append(x)

    output = np.array(output, dtype="float32") 
    return output

def get_refined_sample_indicator(my_order, version = 1, offset = None):
    # we use (x - mean(x))/stderr as the normalization
    # we remove the lattice parameters and G6
    global refined_sample_indicator
    if (type(refined_sample_indicator) == type(None)):
        refined_sample_indicator = np.load(FLAGS.refined_sample_indicator_dir)

    output = []
        
    for i in my_order:
        if (version == 1):
            x = refined_sample_indicator[i][:FLAGS.n_bases + FLAGS.n_new_bases]
        #x = (x - _mean)/_std
        output.append(x)

    output = np.array(output, dtype="float32") 
    return output

def get_refined_shift_indicator(my_order, version = 1, offset = None):
    # we use (x - mean(x))/stderr as the normalization
    # we remove the lattice parameters and G6
    global refined_shift_indicator
    if (type(refined_shift_indicator) == type(None)):
        refined_shift_indicator = np.load(FLAGS.refined_shift_indicator_dir)
    output = []
        
    for i in my_order:
        if (version == 1):
            x = refined_shift_indicator[i]
        #x = (x - _mean)/_std
        output.append(x)

    output = np.array(output, dtype="float32") 
    return output

def get_degree_of_freedom(my_order, offset = None):
    # we use (x - mean(x))/stderr as the normalization
    # we remove the lattice parameters and G6
    output = []
        
    for i in my_order:
        x = degree_of_freedom[i]
        output.append(x)

    output = np.array(output, dtype="float32") 
    return output 
#####################################################################
