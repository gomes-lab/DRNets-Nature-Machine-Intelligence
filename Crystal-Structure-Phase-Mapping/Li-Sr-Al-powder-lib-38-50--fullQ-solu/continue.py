import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import model
import get_data
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score 

FLAGS = tf.app.flags.FLAGS

def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def train_step(epoch, sess, hg, merged_summary, summary_writer, input_feature, input_xrd, train_op, global_step):

	feed_dict={}
	feed_dict[hg.input_feature] = input_feature
	feed_dict[hg.input_xrd] = input_xrd
	feed_dict[hg.keep_prob] = 0.5
	feed_dict[hg.epoch] = epoch

	temp, step, recon_loss_batch, basis_weights, recon_loss, kl_loss, l2_loss, total_loss, summary= \
	sess.run([train_op, global_step, hg.recon_loss_batch, hg.weights, hg.recon_loss, hg.tot_kl_loss, hg.l2_loss, hg.total_loss, merged_summary], feed_dict)

	time_str = datetime.datetime.now().isoformat()
	summary_writer.add_summary(summary,step)

	return recon_loss_batch, basis_weights, recon_loss, kl_loss, l2_loss, total_loss

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

	merged_summary = tf.summary.merge_all() # gather all summary nodes together
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph) #initialize the summary writer

	#sess.run(tf.global_variables_initializer()) # initialize the global variables in tensorflow
	saver = tf.train.Saver(max_to_keep=FLAGS.max_keep) #initializae the model saver
	saver.restore(sess,FLAGS.checkpoint_path)

	print('building finished')

	#initialize several 
	best_loss = 1e10 
	best_iter = 0

	smooth_recon_loss = 0.0
	smooth_kl_loss = 0.0
	smooth_l2_loss = 0.0
	smooth_total_loss = 0.0
	

	for one_epoch in range(FLAGS.max_epoch):
		
		print('epoch '+str(one_epoch+1)+' starts!')

		np.random.shuffle(train_idx) # random shuffle the training indices  
		
		for i in range(train_idx.shape[0]):
			
			indices = batches[train_idx[i]]

			input_feature = get_data.get_feature(data, indices) # get the FEATURE features 
			input_xrd = get_data.get_xrd(data, indices) # get the xrd features 

			recon_loss_batch, basis_weights, recon_loss, kl_loss, l2_loss, total_loss = train_step(one_epoch, sess, hg, merged_summary, summary_writer, input_feature, input_xrd, train_op, global_step)

			smooth_recon_loss += recon_loss
			smooth_kl_loss += kl_loss
			smooth_l2_loss += l2_loss
			smooth_total_loss += total_loss
		
			current_step = sess.run(global_step) #get the value of global_step

			if current_step%FLAGS.check_freq==0: #summarize the current training status and print them out

				recon_loss = smooth_recon_loss / float(FLAGS.check_freq)
				kl_loss = smooth_kl_loss / float(FLAGS.check_freq)
				l2_loss = smooth_l2_loss / float(FLAGS.check_freq)
				total_loss = smooth_total_loss / float(FLAGS.check_freq)
		
				time_str = datetime.datetime.now().isoformat()
				#print out the real-time status of the model  
				print("%s\tstep=%d\trecon_loss=%.6f\tkl_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f\t" % \
					(time_str, current_step, recon_loss, kl_loss, l2_loss, total_loss))

				
				for (k, basis) in enumerate(basis_weights):
					for w in basis:
						print("%3.2f"%w, end = " ")
					print("|\t%3d\trecon = %.3f\t|"%(indices[k], recon_loss_batch[k]), end = " ")

					for w in weights_sol[indices[k]]:
						print("%3.2f"%abs(w), end = " ")
					print("")
				

				smooth_recon_loss = 0
				smooth_kl_loss = 0
				smooth_l2_loss = 0
				smooth_total_loss = 0

			if current_step % FLAGS.valid_freq==0: #exam the model on validation set
				save_model(current_step, saver, sess)


	print('training completed !')
	print('the best loss on validation is '+str(best_loss))
	print('the best checkpoint is '+str(best_iter))
	

if __name__=='__main__':
	tf.app.run()
