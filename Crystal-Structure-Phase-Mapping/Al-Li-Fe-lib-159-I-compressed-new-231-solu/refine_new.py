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
    weights_sol = get_data.get_weights_sol()
    bases_sol = get_data.get_bases_sol()
    decomposition_sol = get_data.get_decomposition_sol()

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
    
    saver = tf.train.Saver(max_to_keep=None)
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
        feed_dict[hg.input_indicator] = get_data.get_refined_sample_indicator(idx)
        feed_dict[hg.shift_indicator] = get_data.get_refined_shift_indicator(idx)
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



    refined_sample_indicator2 = np.zeros_like(np.load(FLAGS.refined_sample_indicator_dir))
    refined_shift_indicator2 = np.zeros_like(np.load(FLAGS.refined_shift_indicator_dir)) + 1

    sum_basis_weights = np.sum(basis_weights * np.greater(basis_weights, 0.1).astype(int), axis = 0)

    act_order = np.argsort(sum_basis_weights)[::-1]

    for i in act_order:
        if (sum_basis_weights[i] > 0.5):
            print(i, sum_basis_weights[i])
        else:
            break
    #stats("bases activation", sum_basis_weights)

    occur = np.zeros(FLAGS.n_bases + FLAGS.n_new_bases)

    for i in range(N):
        w = basis_weights[i]

        idx = np.argsort(w)[::-1]

        cnt = 0

        for j in idx[:3]:
            if (w[j] >= FLAGS.active_th and sum_basis_weights[j] > 0.5):
                cnt +=1
                refined_sample_indicator2[i][j] = 1
                occur[j] += 1
                #if (j == 73):
                #    print(i)
                if (cnt == 3):
                    refined_shift_indicator2[i] = 0
                    break      

    exist_bases = []
    top_bases = np.argsort(occur)[::-1]
    for i in top_bases:
        if (occur[i] > 0):
            print(i, occur[i])
            exist_bases.append(i)
        else:
            break
    np.save("exist_bases", exist_bases)

    np.save(FLAGS.refined_sample_indicator_dir[:-4]+"2", refined_sample_indicator2)
    np.save(FLAGS.refined_shift_indicator_dir[:-4]+"2", refined_shift_indicator2)

if __name__=='__main__':
    tf.app.run()



