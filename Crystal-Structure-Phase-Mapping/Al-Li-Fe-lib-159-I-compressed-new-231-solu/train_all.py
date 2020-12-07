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

def train_step(epoch, sess, hg, merged_summary, summary_writer, input_feature, input_xrd, input_indicator, grad_op, global_step):

	feed_dict={}
	feed_dict[hg.input_feature] = input_feature
	feed_dict[hg.input_xrd] = input_xrd
	feed_dict[hg.input_indicator] = input_indicator
	feed_dict[hg.keep_prob] = FLAGS.keep_prob
	feed_dict[hg.epoch] = epoch

	grad, step, gibbs_loss_batch, recon_loss_batch, basis_weights, recon_loss, kl_loss, l2_loss, total_loss, summary= \
	sess.run([grad_op, global_step, hg.gibbs_loss_batch, hg.recon_loss_batch, hg.weights, hg.recon_loss, hg.tot_kl_loss, hg.l2_loss, hg.total_loss, merged_summary], feed_dict)

	time_str = datetime.datetime.now().isoformat()
	summary_writer.add_summary(summary,step)

	return grad, gibbs_loss_batch, recon_loss_batch, basis_weights, recon_loss, kl_loss, l2_loss, total_loss

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

def average_gradients(tower_grads, grad_op):
    average_grads = []
    cnt = 0
    for grad_and_vars in zip(*tower_grads):

        v = grad_op[cnt][1] #grad_and_vars[0][1]
        #print(v)
        grads = np.zeros_like(grad_and_vars[0][0])
        for g, _ in grad_and_vars:
            grads += g

        grad = grads / len(grad_and_vars)

        #v = tf.constant(v)
        #grad = tf.constant(grad)
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        cnt += 1
    return average_grads

def main(_):

	print('reading npy...')
	np.random.seed(19950420) # set the random seed of numpy 
	data, batches = get_data.get_data() #XRD sources and batches
	train_idx = np.arange(batches.shape[0]) #load the indices of the training set

	one_epoch_iter = train_idx.shape[0] # compute the number of iterations in each epoch

	xrd = get_data.get_xrd_mat()
	composition = get_data.get_comp()
	weights_sol = get_data.get_weights_sol()
	bases_sol = get_data.get_bases_sol()
	decomposition_sol = get_data.get_decomposition_sol()

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
	hg = model.MODEL(is_training=True)

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
		grad_op = optimizer.compute_gradients(hg.optimizer_loss)

	merged_summary = tf.summary.merge_all() # gather all summary nodes together
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph) #initialize the summary writer

	sess.run(tf.global_variables_initializer()) # initialize the global variables in tensorflow
	saver = tf.train.Saver(max_to_keep=FLAGS.max_keep) #initializae the model saver
	if (len(sys.argv) >=2 and "con" in sys.argv[1]):
		print(sys.argv)
		saver.restore(sess,FLAGS.checkpoint_path)
	#saver.restore(sess,FLAGS.checkpoint_path)

	print('building finished')

	#initialize several 
	best_loss = 1e10 
	best_iter = 0

	smooth_recon_loss = 0.0
	smooth_kl_loss = 0.0
	smooth_l2_loss = 0.0
	smooth_total_loss = 0.0
	best_loss = 1e10

	for one_epoch in range(FLAGS.max_epoch):
		
		print('epoch '+str(one_epoch+1)+' starts!')

		np.random.shuffle(train_idx) # random shuffle the training indices  
		grads = []
		for i in range(xrd.shape[0]):
			
			indices = [i]
			#batches[train_idx[i]]

			input_feature = get_data.get_feature(data, indices) # get the FEATURE features 
			input_xrd = get_data.get_xrd(data, indices) # get the xrd features 
			input_indicator = get_data.get_indicator(indices)

			grad, gibbs_loss_batch, recon_loss_batch, basis_weights, recon_loss, kl_loss, l2_loss, total_loss = train_step(one_epoch, sess, hg, merged_summary, summary_writer, input_feature, input_xrd, input_indicator, grad_op, global_step)
			#print("grad", grad[0][0].shape, grad[0][1])

			smooth_recon_loss += recon_loss
			smooth_kl_loss += kl_loss
			smooth_l2_loss += l2_loss
			smooth_total_loss += total_loss
		
			current_step = sess.run(global_step) #get the value of global_step
			grads.append(grad)

		#print(len(grads[0]))
		avg_grads = average_gradients(grads, grad_op)
		#print(len(avg_grads))
		update_op = optimizer.apply_gradients(avg_grads, global_step = global_step)
		sess.run(update_op)

		if (True): #exam the model on validation set
			print("validation at epoch %d"%one_epoch)

			#indices = np.arange()
			st_basis_weights = []
			st_avg_recon_loss = []
			st_recon_loss = []
			st_kl_loss = []
			st_decomp = [] 
			st_xrd_prime  = []
			N = xrd.shape[0]

			for j in range(N):
				#if (j%10 == 0):
				#	print("%.2f%c"%(j*100.0/N, '%'))
				idx = [j]
				feed_dict={}
				feed_dict[hg.input_feature] = get_data.get_feature(data, idx) # get the FEATURE features 
				feed_dict[hg.input_xrd] = get_data.get_xrd(data, idx) 
				feed_dict[hg.input_indicator] = get_data.get_indicator(idx)
				feed_dict[hg.keep_prob] = 1.0
				feed_dict[hg.epoch] = 0.0

				tmp_basis_weights, tmp_avg_recond_loss, tmp_recon_loss, tmp_kl_loss, tmp_decomp, tmp_xrd_prime = \
				sess.run([hg.weights, hg.recon_loss, hg.recon_loss_batch, hg.tot_kl_loss, hg.decomp, hg.xrd_prime], feed_dict)

				st_basis_weights.append(tmp_basis_weights)
				st_avg_recon_loss.append(tmp_avg_recond_loss)
				st_recon_loss.append(tmp_recon_loss)
				st_kl_loss.append(tmp_kl_loss)
				st_decomp.append(tmp_decomp)
				st_xrd_prime.append(tmp_xrd_prime)

			#print("step-1")
			basis_weights = np.concatenate(st_basis_weights, axis = 0)
			avg_recon_loss = np.mean(st_avg_recon_loss)
			recon_loss = np.concatenate(st_recon_loss, axis = 0)
			kl_loss = np.mean(st_kl_loss)
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
			if (np.max(recon_loss) < best_loss):

				best_loss = np.max(recon_loss)
				print("Find better parameters with recon_loss = %.6f"%best_loss)
				
				print("")
				save_model(current_step, saver, sess)
				print("")


	print('training completed !')
	print('the best loss on validation is '+str(best_loss))
	print('the best checkpoint is '+str(best_iter))
	

if __name__=='__main__':
	tf.app.run()
