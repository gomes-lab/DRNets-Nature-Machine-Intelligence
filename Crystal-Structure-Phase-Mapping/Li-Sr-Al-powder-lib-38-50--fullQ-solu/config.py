import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', './model/','path to store the checkpoints of the model')
tf.app.flags.DEFINE_string('summary_dir', './summary/','path to store analysis summaries used for tensorboard')
tf.app.flags.DEFINE_string('config_dir', './config.py','path to config.py')
tf.app.flags.DEFINE_string('checkpoint_path', './model/model-14000','The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('visual_dir', './visualization/','path to store visualization codes and data')

tf.app.flags.DEFINE_string('data_dir', '../data/Li-Sr-Al/data.npy','The path of input observation data')
tf.app.flags.DEFINE_string('batches_dir', '../data/Li-Sr-Al/20k-paths-len5.npy','The path of batches')
#tf.app.flags.DEFINE_string('batches_dir', '../data/paths-len10-balanced.npy','The path of batches')
#tf.app.flags.DEFINE_string('bases_sol_dir', '../data/bases_sol.npy','The path of *')
#tf.app.flags.DEFINE_string('weights_sol_dir', '../data/normalized_weights_sol.npy','The path of *')
#tf.app.flags.DEFINE_string('decompoSrtion_sol_dir', '../data/decompoSrtion_sol.npy','The path of *')
tf.app.flags.DEFINE_string('bases_comp_dir', '../data/Li-Sr-Al/real_lib_comp.npy','The path of *')
tf.app.flags.DEFINE_string('bases_name_dir', '../data/Li-Sr-Al/bases_name.npy','The path of *')
tf.app.flags.DEFINE_string('stick_bases_dir', '../data/Li-Sr-Al/sticks_lib.npy','The path of *')
tf.app.flags.DEFINE_string('bases_edge_dir', '../data/Li-Sr-Al/bases_edge.npy','The path of *')
######
tf.app.flags.DEFINE_string('edges_dir', '../data/Li-Sr-Al/edges.npy','The path of *')
tf.app.flags.DEFINE_string('Q_dir', '../data/Li-Sr-Al/Q.npy','The path of *')
tf.app.flags.DEFINE_string('Q_idx_dir', '../data/Li-Sr-Al/Q_idx.npy','The path of *')
tf.app.flags.DEFINE_string('sample_indicator_dir', '../data/Li-Sr-Al/sample_indicator.npy','The path of *')
tf.app.flags.DEFINE_string('refined_sample_indicator_dir', 'refined_sample_indicator.npy','The path of *')
tf.app.flags.DEFINE_string('refined_shift_indicator_dir', 'refined_shift_indicator.npy','The path of *')
tf.app.flags.DEFINE_string('degree_of_freedom_dir', '../data/Li-Sr-Al/degree_of_freedom.npy','The path of *')
tf.app.flags.DEFINE_string('gt_weights_dir', '../data/Li-Sr-Al/gt_weights.npy','The path of *')

#tf.app.flags.DEFINE_string('train_idx', '../data/train_idx2.npy','The path of training data index')
#tf.app.flags.DEFINE_string('valid_idx', '../data/valid_idx2.npy','The path of validation data index')
#tf.app.flags.DEFINE_string('test_idx', '../data/test_idx2.npy','The path of testing data index')

tf.app.flags.DEFINE_integer('batch_size', 5, 'the number of data points in one minibatch') #128
tf.app.flags.DEFINE_integer('testing_size', 50, 'the number of data points in one testing or validation batch') #128
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate')
tf.app.flags.DEFINE_float('beta', 0.5, 'Focal loss parameter beta')
tf.app.flags.DEFINE_float('alpha', 0.7, 'Focal loss parameter alpha')
tf.app.flags.DEFINE_integer('max_epoch', 200, 'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay', 1e-10, 'weight decay rate')
tf.app.flags.DEFINE_float('similarity_decay', 0.0000, 'weight decay rate of similarity loss')
tf.app.flags.DEFINE_float('gibbs_decay', 30.0, 'weight decay rate of gibbs loss')
tf.app.flags.DEFINE_float('mean_loss', 100.00, 'mean of the recon loss')
tf.app.flags.DEFINE_float('gibbs_th', 1.09, 'threshold of the gibbs loss')
tf.app.flags.DEFINE_float('gibbs_penalty', 0.05, '*')
tf.app.flags.DEFINE_float('alloy_decay', 0.0, '*')
tf.app.flags.DEFINE_float('comp_decay', 0.05, 'weight decay rate of comp loss')
tf.app.flags.DEFINE_float('smoothness_decay', 3.0, 'weight decay rate of *')
tf.app.flags.DEFINE_float('smooth_weights_th', 0.000, 'threshold of *')
tf.app.flags.DEFINE_float('shift_amplify', 1.0, '*')
tf.app.flags.DEFINE_float('shift_unit', 0.1, '*')
tf.app.flags.DEFINE_float('max_shift', 0.04, 'the maximum shift range for sticks')
tf.app.flags.DEFINE_float('min_activation', 0.01, 'the threshold for weights')
tf.app.flags.DEFINE_float('active_th', 0.01, 'the threshold for bases to be considered as active bases')

tf.app.flags.DEFINE_float('KL_weight', 0.20, 'weight decay rate of *')
tf.app.flags.DEFINE_float('L2_weight', 0.05, 'weight decay rate of *')
tf.app.flags.DEFINE_float('L1_weight', 0.00, 'weight decay rate of *')
tf.app.flags.DEFINE_float('intensity_shift', 1.2, '*') # 1.5
tf.app.flags.DEFINE_float('global_intensity_shift', 1.0, '*')
tf.app.flags.DEFINE_float('intensity_shift_loss_decay', 0.001, '*')
tf.app.flags.DEFINE_float('intensity_th', 0.05, '*')
tf.app.flags.DEFINE_float('sqr_recon_loss_decay', 0.0, 'weight decay rate of *')
tf.app.flags.DEFINE_float('threshold', 0.5, 'The probability threshold for the prediction')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'The probability dropout')
tf.app.flags.DEFINE_float('noise_scale', 0.00, '*')
tf.app.flags.DEFINE_float('init_logvar', -5.0, 'The initial logvar for Gaussian Mixtures')
tf.app.flags.DEFINE_float('logvar_range', 0.5, 'The range of logvar for Gaussian Mixtures')
tf.app.flags.DEFINE_float('max_extra_logvar', 1.0, 'The range of max increased logvar for Gaussian Mixtures')
tf.app.flags.DEFINE_float('KL_eps', 1e-6, 'eps')
tf.app.flags.DEFINE_float('eps', 1e-8, 'eps')
tf.app.flags.DEFINE_float('lr_decay_ratio', 1.0, 'The decay ratio of learning rate')
tf.app.flags.DEFINE_float('lr_decay_times', 200.0, 'How many times does learning rate decay')
tf.app.flags.DEFINE_integer('n_test_sample', 1000, 'The sampling times for the testing')
tf.app.flags.DEFINE_integer('n_sample', 1, 'The sampling times for the training') #100
tf.app.flags.DEFINE_integer('rand_I', 0, '1: use random indicator, 0: not use') #100


tf.app.flags.DEFINE_integer('z_dim', 64, '#dimensionality of VAEs')
tf.app.flags.DEFINE_integer('n_spike', 200, '#spikes')

#tf.app.flags.DEFINE_integer('r_offset', 0, 'the offset of labels')
#tf.app.flags.DEFINE_integer('r_dim', 77, 'the number of labels in current training') 
tf.app.flags.DEFINE_integer('n_new_bases', 1, '# new bases')
tf.app.flags.DEFINE_integer('n_bases', 37, '#library bases ')
tf.app.flags.DEFINE_integer('feature_dim', 3, 'the dimensionality of the gemeral features ') 
tf.app.flags.DEFINE_integer('remove_dim', 0, 'the dimensionality of the removed features ') 
tf.app.flags.DEFINE_integer('xrd_dim', 4501, 'the dimensionality of the xrd features ')
tf.app.flags.DEFINE_integer('compressed_xrd_dim', 4501, 'the dimensionality of the compressed xrd features ')
#tf.app.flags.DEFINE_integer('user_dim', 6, 'the dimensionality of the user-features')

tf.app.flags.DEFINE_float('save_epoch', 1.0, 'epochs to save the checkpoint of the model')
tf.app.flags.DEFINE_integer('max_keep', 10, 'maximum number of saved model')
tf.app.flags.DEFINE_integer('check_freq', 10, 'checking frequency')
tf.app.flags.DEFINE_integer('valid_freq', 1000, 'checking frequency')



