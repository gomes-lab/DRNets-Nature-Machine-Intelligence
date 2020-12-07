import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

import numpy as np
import datetime
import model
import get_data
import config 
import os, sys
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score 

FLAGS = tf.app.flags.FLAGS
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

def stats(name, x):
    print(name, x.shape)
    idx = np.argsort(x)
    print("stats of %s min:"%name, np.min(x), "max:", np.max(x), "mean", np.mean(x), "median", np.median(x), "std", np.std(x), "worse_idx", idx[-1])
    return idx

def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def save_model(step, saver, sess):
    print('Saving the model at step-%d'%step)

    saved_model_path = saver.save(sess, FLAGS.model_dir+'model', global_step=step)

    print('have saved model to ', saved_model_path)
    print("rewriting the number of model to config.py")

    #write the best checkpoint number back to the config.py file
    configFile=open(FLAGS.config_dir, "r")
    content=[line.strip("\n") for line in configFile]
    configFile.close()

    for i in range(len(content)):
        if ("checkpoint_path" in content[i]):
            content[i]="tf.app.flags.DEFINE_string('checkpoint_path', './model/model-%d','The path to a checkpoint from which to fine-tune.')"%step
    
    configFile=open(FLAGS.config_dir, "w")
    for line in content:
        configFile.write(line+"\n")
    configFile.close()

def main(_):

    print('reading npy...')
    np.random.seed(19941216) # set the random seed of numpy 
    data, batches = get_data.get_data() #XRD sources and batches
    train_idx = np.arange(batches.shape[0]) #load the indices of the training set

    one_epoch_iter = train_idx.shape[0] # compute the number of iterations in each epoch

    xrd = get_data.get_xrd_mat()
    composition = get_data.get_comp()
    #weights_sol = get_data.get_weights_sol()
    #bases_sol = get_data.get_bases_sol()
    #decomposition_sol = get_data.get_decomposition_sol()

    print('reading completed')

    # config the tensorflow
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    ##debugger
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    print('showing the parameters...\n')


    for key in FLAGS:
        value = FLAGS[key].value
        print("%s\t%s"%(key, value))
    print("\n")


    print('building network...')

    #building the model

    flag = True
    fac = 1.0
    if ("refine" in sys.argv):
        flag = True
        fac = 1.0

    hg = model.MODEL(is_training = flag)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate * fac, global_step, (1.0/FLAGS.lr_decay_times)*(FLAGS.max_epoch*one_epoch_iter), FLAGS.lr_decay_ratio, staircase=True)

    if ("refine" in sys.argv):
        learning_rate /= 3.0

    #log the learning rate 
    tf.summary.scalar('learning_rate', learning_rate)

    #use the Adam optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)

    #set training update ops/backpropagation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    noise_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "noise")
    #print(noise_var)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(hg.optimizer_loss, global_step = global_step)
        grad = optimizer.compute_gradients(hg.optimizer_loss, noise_var)

    merged_summary = tf.summary.merge_all() # gather all summary nodes together
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph) #initialize the summary writer

    sess.run(tf.global_variables_initializer()) # initialize the global variables in tensorflow
    Vars = tf.global_variables()
    Vars_filtered = []
    for v in Vars:
        if (not ("Adam" in v.name and "spike_shift" in v.name)):
            Vars_filtered.append(v)

    saver = tf.train.Saver(max_to_keep=None, var_list = Vars_filtered)
    saver2 = tf.train.Saver(max_to_keep=FLAGS.max_keep) #initializae the model saver2

    if ("con" in sys.argv):
        print(sys.argv)
        saver.restore(sess,FLAGS.checkpoint_path)
    #saver.restore(sess,FLAGS.checkpoint_path)

    print('building finished')

    #initialize several 
    best_loss = 1e10 
    best_iter = 0

    smooth_recon_loss = 0.0
    smooth_l2_loss = 0.0
    smooth_total_loss = 0.0
    best_loss = 1e10
    Round = 0
    base_step = 0 
    pretrain_step = 3000
    st_condition = []

    if ("new" in sys.argv):
        base_step = sess.run(global_step)
        pretrain_step = 3000

    for one_epoch in range(FLAGS.max_epoch):
        
        print('epoch '+str(one_epoch+1)+' starts!')

        np.random.shuffle(train_idx) # random shuffle the training indices  
        
        for i in range(train_idx.shape[0]):
            
            indices = batches[train_idx[i]]
            #[167, 91, 38, 108, 66, 153, 67, 268, 113, 129] 
            #[303, 298, 312, 306, 301, 296, 309, 304, 299, 313] 

            input_feature = get_data.get_feature(data, indices) # get the FEATURE features 
            input_xrd = get_data.get_xrd(data, indices) # get the xrd features 
           
            
            #if ("con" in sys.argv and not("new" in sys.argv)):
            #    Round = 10.0

            
            if ("refine" in sys.argv):
                input_indicator = get_data.get_refined_sample_indicator(indices)
                shift_indicator = get_data.get_refined_shift_indicator(indices)
            else:
                input_indicator = get_data.get_indicator(indices)
                shift_indicator = np.zeros((input_xrd.shape[0], 1)) + 1
             
            current_step = sess.run(global_step) #get the value of global_step
            Round = max(0, (current_step - base_step - pretrain_step) / 300.0)
            
            feed_dict={}
            feed_dict[hg.input_feature] = input_feature
            feed_dict[hg.input_xrd] = input_xrd
            feed_dict[hg.input_indicator] = input_indicator
            feed_dict[hg.shift_indicator] = shift_indicator
            feed_dict[hg.degree_of_freedom] = get_data.get_degree_of_freedom(indices)
            feed_dict[hg.keep_prob] = FLAGS.keep_prob
            feed_dict[hg.epoch] = Round


            temp, step, comp, comp_prime, noise, noise_b, c1, c2, c3, gibbs_penalty_ratio, gibbs_loss_batch, max_shift, alloy_loss_batch, recon_loss_batch, basis_weights, recon_loss, l2_loss, total_loss, summary= \
            sess.run([train_op, global_step, hg.comp, hg.comp_prime, hg.noise, hg.noise_b, hg.condition1, hg.condition2, hg.condition3, hg.gibbs_penalty_ratio, hg.gibbs_loss_batch, hg.max_shift, hg.alloy_loss_batch, hg.recon_loss_batch, hg.weights, hg.recon_loss, hg.l2_loss, hg.total_loss, merged_summary], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            summary_writer.add_summary(summary,step)


            smooth_recon_loss += recon_loss
            smooth_l2_loss += l2_loss
            smooth_total_loss += total_loss
            st_condition.append(np.logical_or(c1, np.logical_and(c2, c3))) 
            current_step = sess.run(global_step) #get the value of global_step
            

            if current_step%FLAGS.check_freq==0: #summarize the current training status and print them out
                recon_loss = smooth_recon_loss / float(FLAGS.check_freq)
                l2_loss = smooth_l2_loss / float(FLAGS.check_freq)
                total_loss = smooth_total_loss / float(FLAGS.check_freq)
                
                condition = np.concatenate(st_condition, axis = 0)

                summary_writer.add_summary(MakeSummary('train/percent_of_violation', np.mean(condition.astype("float32"))), current_step)


                time_str = datetime.datetime.now().isoformat()
                #print out the real-time status of the model  
                print("%s\tstep=%d\trecon_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f\t" % \
                    (time_str, current_step, recon_loss, l2_loss, total_loss))

                ans_set = [[14], [36], [7]]
                for (k, basis_w) in enumerate(basis_weights):
                    #sum_w = 0
                    #for z in ans_set:
                    #   sum_w += basis_w[z]
                    for z in ans_set:
                        act = np.sum(basis_w[z])
                        if (np.sum(input_indicator[k][z]) >= 1):
                            print("%3.2f"%(act), end = " ")
                        else:
                            print(color.GREY+"%3.2f"%(act)+color.END, end = " ")
    
                    s = color.GREEN
                    
                    #if (indices[k] in [34, 47]):
                    #    s = color.RED
                    print(s+"%3.2f"%(basis_w[-1])+color.END, end = " ")
                    #for w in basis_w[:9]:
                    #   print("%3.2f"%w, end = " ")
                    idx_str = "%3d"%indices[k]
                    print("| %s| recon = %7.3f| gibbs = %.3f| max_shift = %.6f| alloy = %6.3f| noise = %.3f| r = %.1f|"%(idx_str, recon_loss_batch[k], gibbs_loss_batch[k], max_shift[k], alloy_loss_batch[k], np.mean(noise), gibbs_penalty_ratio[k]), end = " ")

                    #sum_w = 0
                    #for w in weights_sol[indices[k]]:
                    #   sum_w += w
                    #print(" | ", input_indicator[k], end = " ")
                    #print("\t| ", input_feature[k])
                    print("| c1 = %d| c2 = %d| c3 =%d|"%(c1[k], c2[k], c3[k]), end = " ")
                    #print("|", end = "")
                
                    for c in comp[k]:
                        print("%.2f"%c, end = " ")

                    print("|", end = " ")
                    for c in comp_prime[k]:
                        print("%.2f"%c, end = " ")
                    
                    print("")
                

                smooth_recon_loss = 0
                smooth_l2_loss = 0
                smooth_total_loss = 0
                st_condition = []
            
            if current_step % FLAGS.valid_freq==0: #exam the model on validation set
                print("validation at epoch %d"%current_step) 
                #indices = np.arange()
                st_basis_weights = []
                st_avg_recon_loss = []
                st_recon_loss = []
                st_decomp = [] 
                st_xrd_prime  = []
                N = xrd.shape[0]

                for j in range(N):
                    #if (j%10 == 0):
                    #   print("%.2f%c"%(j*100.0/N, '%'))
                    idx = [j]
                    feed_dict={}
                    feed_dict[hg.input_feature] = get_data.get_feature(data, idx) # get the FEATURE features 
                    feed_dict[hg.input_xrd] = get_data.get_xrd(data, idx) 

                    if ("refine" in sys.argv):
                        feed_dict[hg.input_indicator] = get_data.get_refined_sample_indicator(idx)
                        feed_dict[hg.shift_indicator] = get_data.get_refined_shift_indicator(idx)
                    else:
                        feed_dict[hg.input_indicator] = get_data.get_indicator(idx)
                        feed_dict[hg.shift_indicator] = np.zeros((input_xrd.shape[0], 1)) + 1

                    feed_dict[hg.degree_of_freedom] = get_data.get_degree_of_freedom(idx)
                    feed_dict[hg.keep_prob] = 1.0
                    feed_dict[hg.epoch] = 10.0

                    tmp_basis_weights, tmp_avg_recond_loss, tmp_recon_loss, tmp_decomp, tmp_xrd_prime = \
                    sess.run([hg.weights, hg.recon_loss, hg.recon_loss_batch, hg.decomp, hg.xrd_prime], feed_dict)

                    st_basis_weights.append(tmp_basis_weights)
                    st_avg_recon_loss.append(tmp_avg_recond_loss)
                    st_recon_loss.append(tmp_recon_loss)
                    st_decomp.append(tmp_decomp)
                    st_xrd_prime.append(tmp_xrd_prime)

                #print("step-1")
                basis_weights = np.concatenate(st_basis_weights, axis = 0)
                avg_recon_loss = np.mean(st_avg_recon_loss)
                recon_loss = np.concatenate(st_recon_loss, axis = 0)
                decomp = np.concatenate(st_decomp, axis = 0)
                xrd_prime  = np.concatenate(st_xrd_prime, axis = 0)
                #print("step-2")

                #print("xrd_prime", xrd_prime.shape)
                #idx1 = stats("recon_loss", recon_loss)
                #weights_L1_loss = np.sum(np.abs(basis_weights[:,:6] - weights_sol), axis = 1)
                #idx2 = stats("weights_L1_loss", )
                stats("recon_loss", recon_loss)
                #stats("weights_L1_loss", weights_L1_loss)

                summary_writer.add_summary(MakeSummary('valid/max_recon_loss', np.max(recon_loss)), current_step)
                summary_writer.add_summary(MakeSummary('valid/avg_recon_loss', avg_recon_loss), current_step)
                #summary_writer.add_summary(MakeSummary('valid/avg_weights_L1_loss', np.mean(weights_L1_loss)), current_step)
                if (current_step >= 3000+ pretrain_step + base_step and best_loss > np.max(recon_loss)):

                    best_loss = np.max(recon_loss)
                    print("Find better parameters with recon_loss = %.6f"%best_loss)
                    
                    print("")
                    save_model(current_step, saver, sess)
                    print("")
                else:
                    save_model(current_step, saver2, sess)


    print('training completed !')
    print('the best loss on validation is '+str(best_loss))
    print('the best checkpoint is '+str(best_iter))
    

if __name__=='__main__':
    tf.app.run()
