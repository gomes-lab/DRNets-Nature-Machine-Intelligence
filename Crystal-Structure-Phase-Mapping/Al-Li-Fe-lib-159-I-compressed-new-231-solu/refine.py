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
import math
import urllib
import sys
#from pyheatmap.heatmap import HeatMap
#import seaborn as sns

FLAGS = tf.app.flags.FLAGS
def stats(name, x, visual = True):
    idx = np.argsort(x)
    print("stats of %s min:"%name, np.min(x), "max:", np.max(x), "mean", np.mean(x), "median", np.median(x), "std", np.std(x), "worse_idx", idx[-1])
    if (visual):
        plt.title(name)
        plt.hist(x, 100)
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
    vec1 = np.array([0,0])
    vec2 = np.array([1.0/2, np.sqrt(3.0/4)])
    vec3 = np.array([1,0])
    array = []


    for i in range(len(comp)):
        array.append(comp[i][0]*vec1 + comp[i][1]*vec2 + comp[i][2]*vec3)

    return np.asarray(array) # (x_i, y_i)
    #x = [item[0] for item in array]
    #y = [item[1] for item in array]

    #return x,y

def main(_):

    print('reading npy...')
    np.random.seed(19950420) # set the random seed of numpy 
    data, batches = get_data.get_data() #XRD sources and batches
    train_idx = np.arange(batches.shape[0]) #load the indices of the training set

    xrd = get_data.get_xrd_mat()
    composition = get_data.get_comp()
    degree_of_freedom = get_data.get_degree_of_freedom(np.arange(FLAGS.testing_size))
    #max_peak = np.max(xrd, axis=1)
    #plt.hist(max_peak)
    #plt.show()
    #visual_bases_sol(bases_sol)

    one_epoch_iter = train_idx.shape[0] # compute the number of iterations in each epoch

    print('reading completed')

    # config the tensorflow
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    print('showing the parameters...\n')

    for key in FLAGS:
        value = FLAGS[key].value
        print("%s\t%s"%(key, value))
    print("\n")


    print('building network...')

    #building the model 
    hg = model.MODEL(is_training=False)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, (1.0/FLAGS.lr_decay_times)*(FLAGS.max_epoch*one_epoch_iter), FLAGS.lr_decay_ratio, staircase=True)

    #log the learning rate 
    tf.summary.scalar('learning_rate', learning_rate)

    #use the Adam optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)

    #set training update ops/backpropagation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(hg.optimizer_loss, global_step = global_step)

    #merged_summary = tf.summary.merge_all() # gather all summary nodes together
    #summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph) #initialize the summary writer

    sess.run(tf.global_variables_initializer()) # initialize the global variables in tensorflow
    
    Vars = tf.global_variables()
    Vars_filtered = []
    for v in Vars:
        if (not ("Adam" in v.name and "spike_shift" in v.name)):
            Vars_filtered.append(v)

    saver = tf.train.Saver(max_to_keep=None, var_list = Vars_filtered)
    saver.restore(sess, FLAGS.checkpoint_path)

    print('restoring from '+FLAGS.checkpoint_path)


    print('Testing...')
    N = FLAGS.testing_size
    M = FLAGS.n_bases

    st_basis_weights = []
    st_avg_recon_loss = []
    st_recon_loss = []
    st_decomp = [] 
    st_xrd_prime  = []
    st_mu = []
    st_mu_shift = []
    st_logvar = []
    st_intensity = []
    st_gibbs_loss_batch = []
    for j in range(N):
        #if (j%10 == 0):
        #   print("%.2f%c"%(j*100.0/N, '%'))
        idx = [j]
        feed_dict={}
        feed_dict[hg.input_feature] = get_data.get_feature(data, idx) # get the FEATURE features 
        feed_dict[hg.input_xrd] = get_data.get_xrd(data, idx) 
        feed_dict[hg.input_indicator] = get_data.get_indicator(idx)
        feed_dict[hg.shift_indicator] = np.zeros((len(idx), 1)) + 1
        feed_dict[hg.degree_of_freedom] = get_data.get_degree_of_freedom(idx)
        feed_dict[hg.keep_prob] = 1.0
        feed_dict[hg.epoch] = 10.0

        noise, tmp_gibbs_loss_batch, tmp_basis_weights, tmp_avg_recond_loss, tmp_recon_loss, tmp_decomp, tmp_xrd_prime, tmp_mu, tmp_mu_shift, tmp_logvar, tmp_intensity = \
        sess.run([hg.noise, hg.gibbs_loss_batch, hg.weights, hg.recon_loss, hg.recon_loss_batch,hg.decomp, hg.xrd_prime, hg.mu, hg.mu_shift, hg.logvar, hg.intensity], feed_dict)

        st_gibbs_loss_batch.append(tmp_gibbs_loss_batch)
        st_basis_weights.append(tmp_basis_weights)
        st_avg_recon_loss.append(tmp_avg_recond_loss)
        st_recon_loss.append(tmp_recon_loss)
        st_decomp.append(tmp_decomp)
        st_xrd_prime.append(tmp_xrd_prime)
        st_mu.append(tmp_mu)
        st_mu_shift.append(tmp_mu_shift)
        st_logvar.append(tmp_logvar)
        st_intensity.append(tmp_intensity)

    gibbs_loss_batch = np.concatenate(st_gibbs_loss_batch, axis = 0)
    basis_weights = np.concatenate(st_basis_weights, axis = 0)
    avg_recon_loss = np.mean(st_avg_recon_loss)
    recon_loss = np.concatenate(st_recon_loss, axis = 0)
    decomp = np.concatenate(st_decomp, axis = 0)
    xrd_prime  = np.concatenate(st_xrd_prime, axis = 0)
    mu = np.concatenate(st_mu, axis = 0)
    mu_shift = np.concatenate(st_mu_shift, axis = 0)
    logvar = np.concatenate(st_logvar, axis = 0)
    intensity = np.concatenate(st_intensity, axis = 0)

    sample_indicator = np.load(FLAGS.sample_indicator_dir)

    refined_sample_indicator = np.zeros((sample_indicator.shape[0], FLAGS.n_bases + FLAGS.n_new_bases))
    refined_shift_indicator = np.zeros((N,1)) + 1.0

    edges = np.load(FLAGS.edges_dir)
    sum_basis_weights = np.sum(basis_weights * np.greater(basis_weights, FLAGS.active_th).astype(int), axis = 0)

    act_order = np.argsort(sum_basis_weights)[::-1]

    for i in act_order:
        if (sum_basis_weights[i] > 0.1):
            print(i, sum_basis_weights[i])
        else:
            break
    #stats("bases activation", sum_basis_weights)

    occur = np.zeros(FLAGS.n_bases + FLAGS.n_new_bases)

    for i in range(N):
        w = basis_weights[i]
        
        for j in range(FLAGS.n_bases):
            w[j] *= sample_indicator[i][j]

        idx = np.argsort(w)[::-1]

        is_shifting = 0 

        for j in idx[:int(degree_of_freedom[i])]:
            if (w[j] >= FLAGS.active_th and sum_basis_weights[j] > 0.1):

                refined_sample_indicator[i][j] = 1
        
   
    #refined_sample_indicator = np.load("refined_sample_indicator.npy")

    n_shift = 0
    max_shift = 0
    shift_status = []
    max_shifts = []

    order = np.argsort(np.min(basis_weights * refined_sample_indicator + 1 - refined_sample_indicator, axis = 1))

    for i in order:
        w = basis_weights[i]
        
        for j in range(FLAGS.n_bases):
            w[j] *= sample_indicator[i][j]

        idx = np.argsort(w)[::-1]

        cnt = 0
        is_shifting = 0 
        max_shift = 0
        for j in range(edges.shape[1]):
            if (edges[i][j] == 1):
                n_fi = np.sum(np.greater(composition[i], 1e-6).astype("int"))
                n_fj = np.sum(np.greater(composition[j], 1e-6).astype("int"))
                is_boundary = 1
                if (n_fi == n_fj and np.sum(np.abs(refined_sample_indicator[i] - refined_sample_indicator[j])) == 0):
                    is_boundary = 0

                tmp = 0 
                for k in range(FLAGS.n_bases):
                    if (refined_sample_indicator[i][k] == 1 and refined_sample_indicator[j][k] == 1 and abs(mu_shift[i][k] - mu_shift[j][k]) > FLAGS.shift_unit):
                        tmp = 1
                        max_shift = max(max_shift, abs(mu_shift[i][k] - mu_shift[j][k]))

                if (is_boundary and (np.sum(refined_sample_indicator[i]) < n_fi or np.sum(refined_sample_indicator[j]) < n_fj)):
                    tmp = 0

                if (tmp):
                    is_shifting = 1


        shift_status.append(is_shifting)
        max_shifts.append(max_shift)
        n_shift += is_shifting

        for j in idx[:int(degree_of_freedom[i])]:
            if (w[j] >= FLAGS.active_th and sum_basis_weights[j] > 0.1):

                if (cnt == degree_of_freedom[i] - 1 and is_shifting == 1):
                    refined_sample_indicator[i][j] = 0
                else:
                    refined_sample_indicator[i][j] = 1
                    occur[j] += 1
                cnt +=1
    
    np.save("shift_status", shift_status)
    np.save("max_shifts", max_shifts)
    print("%d points are shifting"%n_shift)
    print("max shift = ", np.max(max_shifts))
    exist_bases = []
    top_bases = np.argsort(occur)[::-1]
    for i in top_bases:
        if (occur[i] > 0):
            print(i, occur[i])
            exist_bases.append(i)
        else:
            break
    np.save("exist_bases", exist_bases)
    #print("top_bases", top_bases)
    #refined_sample_indicator[105][11] = 0
    #refined_sample_indicator[106][11] = 0
    stats("refined_sample_indicator", np.sum(refined_sample_indicator[:, :FLAGS.n_bases + FLAGS.n_new_bases], axis = 1))
    stats("refined_shift_indicator", np.sum(refined_shift_indicator, axis = 1))

    np.save(FLAGS.refined_sample_indicator_dir[:-4], refined_sample_indicator)
    np.save(FLAGS.refined_shift_indicator_dir[:-4], refined_shift_indicator)

    #plt.show()

    
if __name__=='__main__':
    tf.app.run()



