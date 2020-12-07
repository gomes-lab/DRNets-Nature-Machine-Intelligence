import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS
import sys 

def MLP(input_x, output_size, name, weights_regularizer):
    with tf.variable_scope(name): 
        x = slim.fully_connected(input_x, 1024, weights_regularizer = weights_regularizer, scope = 'fc_1')
        x = slim.fully_connected(x, 1024, weights_regularizer = weights_regularizer, scope = 'fc_2')
        x = slim.fully_connected(x, 512, weights_regularizer = weights_regularizer, scope = 'fc_3')
        x = slim.fully_connected(x, output_size, activation_fn = None, weights_regularizer = weights_regularizer, scope = 'logits')
    return x
        
def decoder_net(input_x, weights_regularizer, epoch, shift_indicator, is_training):
    # initialize the basis with a known stick pattern
    #lim = 10.0
    Q = np.load(FLAGS.Q_dir)
    stick_bases = np.load(FLAGS.stick_bases_dir)

    mu_init = np.zeros([FLAGS.n_bases, FLAGS.n_spike])
    c_init = np.zeros([FLAGS.n_bases, FLAGS.n_spike])
    #logvar_init = np.zeros([FLAGS.n_bases, FLAGS.n_spike]) + FLAGS.init_logvar

    for num in range(FLAGS.n_bases):
        basis = stick_bases[num]
        for (i, peak) in enumerate(basis):
            mu_init[num][i] = peak[0] 
            c_init[num][i] = peak[1]
        
        c_init[num] /= np.max(c_init[num]) + FLAGS.eps

    #
    X = Q
    X = tf.reshape(tf.constant(X, dtype= "float32"), [-1, 1, 1, 1]) # X, 1, 1
    X = tf.tile(X, [1, tf.shape(input_x)[0], FLAGS.n_bases + FLAGS.n_new_bases, FLAGS.n_spike]) # Q.size, bs, n_bases, n_spike
    #print(X.shape)
    #mu_init = np.random.uniform(-lim, lim, size = FLAGS.n_spike)
    mu_new_bases = tf.exp(tf.Variable(np.log(np.random.rand(FLAGS.n_new_bases, FLAGS.n_spike)*(Q[-1] - Q[0]) + Q[0]), trainable = True, dtype = "float32"))
    c_new_bases = tf.abs(tf.Variable(np.random.rand(FLAGS.n_new_bases, FLAGS.n_spike), trainable = True, dtype = "float32"))

    mu_init = tf.concat([tf.Variable(mu_init, trainable = False, dtype="float32"), mu_new_bases], axis = 0) #n_bases + n_new_bases, n_spike

    c_init = tf.concat([tf.Variable(c_init, trainable = False, dtype="float32"), c_new_bases], axis = 0) #n_bases + n_new_bases, n_spike

    shift_indicator = tf.reshape(shift_indicator, [-1, 1, 1])
    shift_indicator = tf.tile(shift_indicator, [1, FLAGS.n_bases + FLAGS.n_new_bases, FLAGS.n_spike])

    with tf.variable_scope('decoder'):

        ################ mu
        mu_shift = MLP(input_x, FLAGS.n_bases + FLAGS.n_new_bases, "spike_shift", weights_regularizer) 
        r1 = (tf.minimum(epoch, 10.0) /10.0)
        #alpha = FLAGS.max_shift 
        alpha = r1 * FLAGS.max_shift 
        if ("refine" in sys.argv):
            mu_shift = tf.stop_gradient(tf.tanh(mu_shift)* alpha + 1.0)
        else:
            mu_shift = tf.tanh(mu_shift)* alpha + 1.0

        mu_shift = tf.reshape(mu_shift, [-1, FLAGS.n_bases + FLAGS.n_new_bases, 1])
        mu_shift = tf.tile(mu_shift, [1, 1, FLAGS.n_spike]) #bs, n_bases, n_spike

        #mu_shift = (shift_indicator * mu_shift + (1.0 - shift_indicator) * tf.clip_by_value(mu_shift, 1.0 - FLAGS.shift_unit, 1.0 + FLAGS.shift_unit))
        mu = mu_init *  mu_shift#bs, n_bases, n_spike
        #mu = tf.clip_by_value(mu, 11, 60)
        ################ variance

        logvar_shift = MLP(input_x, FLAGS.n_bases + FLAGS.n_new_bases, "spike_logvar", weights_regularizer) 

        #logvar_shift = tf.Variable(np.zeros((1, FLAGS.n_bases + FLAGS.n_new_bases)), trainable = True, dtype = tf.float32, name = "logvar_shift")
        
        logvar_shift = tf.tanh(logvar_shift) * FLAGS.logvar_range
    
        #logvar_shift = tf.tile(logvar_shift, [tf.shape(mu_shift)[0], 1]) #bs, n_bases

        logvar = logvar_shift + tf.Variable(np.zeros(FLAGS.n_bases + FLAGS.n_new_bases) + FLAGS.init_logvar, trainable = False, dtype = tf.float32, name = "logvar") #bs, n_bases
        
        logvar = tf.reshape(logvar, [-1, FLAGS.n_bases + FLAGS.n_new_bases, 1]) #bs, n_bases, 1

        
        extra_logvar = tf.tanh(tf.abs(tf.Variable(0.1, dtype = tf.float32, trainable = True, name = "gradual_fattening_ratio"))) * FLAGS.max_extra_logvar

        gradual_fattening = mu_init / Q[-1] * extra_logvar # bs, n_bases, n_spike: 0~1
         
        if (is_training):
            tf.summary.scalar("extra_logvar", extra_logvar)
        
        logvar = tf.tile(logvar, [1, 1, FLAGS.n_spike]) + gradual_fattening

        #logvar = FLAGS.init_logvar
        ################ intensity
        c = tf.reshape(c_init, [1, FLAGS.n_bases + FLAGS.n_new_bases, FLAGS.n_spike])  
        
        #lim = tf.minimum(epoch, 10.0) /10.0 * np.log(FLAGS.intensity_shift)
        #r2 = (tf.minimum(epoch, 20.0) /20.0)
        #lim = 0.0 * (1-r2) + r2 * np.log(FLAGS.intensity_shift)
        lim = np.log(FLAGS.intensity_shift)
        if ("refine" in sys.argv):
            lim = np.log(FLAGS.intensity_shift * 2.0)

        with tf.variable_scope("intensity_shift_net"): 
            x = slim.fully_connected(input_x, 512, weights_regularizer = weights_regularizer, scope = 'fc_1')
            x = slim.fully_connected(x, 512, weights_regularizer = weights_regularizer, scope = 'fc_2')
            x = slim.fully_connected(x, 32, weights_regularizer = weights_regularizer, scope = 'fc_3')
            x = slim.fully_connected(x, (FLAGS.n_bases + FLAGS.n_new_bases) * FLAGS.n_spike, activation_fn = None, weights_regularizer = weights_regularizer, scope = 'logits')

        local_intensity_shift = tf.tanh(tf.reshape(x, [-1, FLAGS.n_bases + FLAGS.n_new_bases, FLAGS.n_spike])) * lim
        global_intensity_shift = tf.tanh(tf.Variable(np.zeros((FLAGS.n_bases + FLAGS.n_new_bases, FLAGS.n_spike)), trainable = True, dtype = tf.float32, name = "global_IS")) * np.log(FLAGS.global_intensity_shift) 
        intensity_shift = tf.exp(local_intensity_shift + global_intensity_shift)        #intensity_shift = tf.exp(tf.clip_by_value(tf.Variable(np.zeros((FLAGS.n_bases + FLAGS.n_new_bases, FLAGS.n_spike)), trainable = True, dtype = tf.float32, name = "I_shift"), -lim, lim)) 
        c = tf.tile(c, [tf.shape(input_x)[0], 1, 1]) 

        intensity_shift_weighted = tf.reduce_sum(tf.abs(tf.log(intensity_shift)) * c, axis = 2) #bs, n_bases

        c_shifted = c * intensity_shift ##bs, n_bases, n_spike

        ################ Gaussian Mixtures
        x = tf.exp( - ((X - mu)**2)/(2.0 * tf.exp(logvar)) - logvar * 0.5) * c_shifted  #Gaussian  xrd, bs, n_bases, n_spike
        #x = tf.exp( - (tf.abs(X - mu))/(tf.exp(logvar) + FLAGS.eps) - logvar) * c_shifted   #Laplace   xrd, bs, n_bases, n_spike
        x = tf.reduce_sum(x, axis = 3) #xrd, bs, n_bases
        x = x / (tf.reduce_max(x, axis = 0) + FLAGS.eps) #xrd, bs, n_bases


        

    return x, mu, mu_shift, logvar, c, intensity_shift, intensity_shift_weighted

def gaussian_KL(recog_mu, recog_logvar, prior_mu, prior_logvar): 
    # KL divergence
    KL = - 0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.pow(prior_mu - recog_mu, 2) / (tf.exp(prior_logvar) + FLAGS.eps)
                               - tf.exp(recog_logvar) / (tf.exp(prior_logvar) + FLAGS.eps), axis = 1)
    return KL

def avg_n(x):
    return tf.reduce_mean(tf.stack(x, axis=0), axis=0)

def sum_normalize(x):
    return tf.transpose(tf.transpose(x) / (tf.reduce_sum(x, axis = 1) + FLAGS.eps))

def max_normalize(x):
    return tf.transpose(tf.transpose(x) / (tf.reduce_max(x, axis = 1) + FLAGS.eps))

class VAE:
    def __init__(self, epoch, input_xrd, input_feature, input_indicator, distance, weights_regularizer, shift_indicator, is_training):
        #tf.set_random_seed(19950420)
        z_dim = FLAGS.z_dim
        n_sample = FLAGS.n_sample
        name = "VAE"
        with tf.variable_scope(name):

            ############## Q(z|x) ###############

            x = tf.concat([input_feature, input_xrd], 1)
            self.bases, self.mu, self.mu_shift, self.logvar, self.intensity, self.intensity_shift, self.intensity_shift_weighted = decoder_net(x, weights_regularizer, epoch, shift_indicator, is_training) # all positive #xrd, bs, n_bases
         

def KL_dis(input_xrd, xrd_prime, eps = None, importance = None):
    
    if (eps == None):
        eps = FLAGS.KL_eps
    
    #P = input_xrd
    input_xrd += FLAGS.eps
    P = tf.transpose(tf.transpose(input_xrd) / (tf.reduce_sum(input_xrd, axis = 1) + FLAGS.eps))
    #Q = xrd_prime 
    xrd_prime += FLAGS.eps
    Q = tf.transpose(tf.transpose(xrd_prime) / (tf.reduce_sum(xrd_prime, axis = 1) + FLAGS.eps))
    
    if (importance == None):
        importance = 1.0 #tf.constant(np.load(FLAGS.importance_dir), dtype = tf.float32)

    tmp = P * tf.log((P + eps)/(Q + eps)) * importance  
    res = tf.reduce_sum(tmp, axis = 1) - (tf.reduce_sum(input_xrd, axis = 1) - tf.reduce_sum(xrd_prime, axis = 1)) * 0
    res = res * 100
    return res

def norm2(x):
    return tf.sqrt(FLAGS.eps + tf.reduce_sum(tf.square(x), axis = 1))

def norm1(x):
    return tf.reduce_sum(tf.abs(x), axis = 1)

def L1_dis(x, y):
    return norm1(x - y) * 100 * FLAGS.L1_weight

def L2_dis(x, y):
    return norm2(x - y) * 100 * FLAGS.L2_weight

def JS_dis(input_xrd, xrd_prime, fac):
    
    return  (0.5 * KL_dis(input_xrd, xrd_prime, FLAGS.KL_eps * fac) + 0.5 * KL_dis(xrd_prime, input_xrd, FLAGS.KL_eps * fac)) * FLAGS.KL_weight
   
class MODEL:

    def __init__(self, is_training):

        tf.set_random_seed(19950420)
        #batch_size = FLAGS.batch_size
        #if (not is_training):
        #   batch_size = FLAGS.testing_size

        self.input_feature = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.feature_dim], name='input_feature')
        
        self.input_xrd = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.compressed_xrd_dim], name='input_xrd') # 4096
        
        self.input_indicator = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.n_bases + FLAGS.n_new_bases], name='input_indicator') 

        self.shift_indicator = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='shift_indicator') 

        self.degree_of_freedom = tf.placeholder(dtype=tf.float32, shape=[None], name='degree_of_freedom') 

        self.keep_prob = tf.placeholder(tf.float32) #keep probability for the dropout

        self.epoch = tf.placeholder(tf.float32)

        weights_regularizer = slim.l2_regularizer(FLAGS.weight_decay)

        ############## feature extractor ###############
        self.prev_feature = tf.slice(self.input_feature, [0, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.feature_dim])
        self.next_feature = tf.slice(self.input_feature, [1, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.feature_dim])
        self.distance = norm1(self.prev_feature - self.next_feature)
        
        #self.kl_losses = []
        #self.similarity_losses = []
        ### normalize and denoise
        self.normalized_input_xrd = max_normalize(self.input_xrd)
        #self.normalized_input_xrd = max_normalize(tf.maximum(0.0, self.normalized_input_xrd - FLAGS.intensity_th))
        

        self.VAE = VAE(self.epoch, self.normalized_input_xrd, self.input_feature, self.input_indicator, self.distance, weights_regularizer, self.shift_indicator, is_training)
       
        self.bases = self.VAE.bases #xrd, bs, n_bases
        self.mu = self.VAE.mu
        self.mu_shift = tf.reduce_mean(self.VAE.mu_shift, axis = 2) # bs, n_bases
        self.logvar = tf.reduce_mean(self.VAE.logvar, axis = 2) # bs, n_bases
        self.intensity = self.VAE.intensity
        self.intensity_shift = self.VAE.intensity_shift  #bs, n_bases, n_sticks


        tf.summary.histogram("intensity_shift", tf.reshape(self.intensity_shift, [-1]))
        tf.summary.histogram("mu_shift", tf.reshape(self.mu_shift, [-1]))

        ##### compute weights for each basis ######
        
        x = tf.concat([self.normalized_input_xrd, self.input_feature], axis = 1)
        x = MLP(x, FLAGS.n_bases + FLAGS.n_new_bases, "classifier", weights_regularizer)
        x = tf.transpose(tf.transpose(x) - tf.reduce_max(x, axis = 1))
        #self.amplifier = tf.exp(x) 
        #self.partition = tf.reduce_sum(self.amplifier, axis = 1)
        
        s = slim.dropout(tf.exp(x), keep_prob = self.keep_prob, is_training = is_training) 
        #s = tf.nn.sigmoid(x) 

        s = s * self.input_indicator #
        sum_s = tf.reduce_sum(s, axis = 1) + FLAGS.eps #bs
        
        #self.weights = s
        self.weights = tf.transpose(tf.transpose(s) / sum_s)

        #### intensity_shift_loss ###

        self.intensity_shift_loss = tf.reduce_mean(tf.reduce_sum(self.VAE.intensity_shift_weighted * tf.stop_gradient(self.weights), axis = 1))
        
        if (is_training):
            tf.summary.scalar("train/intensity_shift_loss", self.intensity_shift_loss)
    
        ##### reconstruction loss ######
        tmp = self.bases * self.weights #xrd, bs, n_bases
        
        self.decomp = tf.transpose(tmp, perm = [1, 2, 0]) # bs, n_bases, xrd

        tmp2 = tf.reduce_sum(tmp, axis=2) # xrd, bs

        noise = tf.abs(tf.Variable(np.random.randn(FLAGS.compressed_xrd_dim)/100.0, trainable = True, dtype="float32", name = "noise")) 
        noise = noise / (tf.reduce_max(noise) + FLAGS.eps)

        scale = tf.nn.sigmoid(tf.Variable(-2, dtype="float32", name = "noise_scale")) * FLAGS.noise_scale
        self.noise = noise * scale 

        #####
        x = tf.concat([self.normalized_input_xrd, self.input_feature], axis = 1)
        x = MLP(x, 1, "noise_b", weights_regularizer)
        self.noise_b = tf.abs(tf.tile(x, [1, FLAGS.compressed_xrd_dim])) * 0.0
        #####

        self.xrd_prime = max_normalize(tf.transpose(tmp2)) + self.noise + self.noise_b
        #self.xrd_prime = tf.transpose(tmp2)  + self.noise

        ###### composition Loss ##############
        max_I_XRD = tf.reduce_max(self.input_xrd, axis = 1)
        max_I_x_prime = tf.reduce_max(tmp2, axis = 0) + FLAGS.eps
        ratio = max_I_XRD / max_I_x_prime

        self.activation = tf.transpose(tf.transpose(self.weights) * ratio) # bs, n_bases
        self.rescale_factor = tf.nn.sigmoid(tf.Variable(np.zeros(FLAGS.n_bases + FLAGS.n_new_bases), trainable = True, dtype="float32", name = "rescale_factor"))

        self.rescaled_activation = self.activation * self.rescale_factor

        self.comp = self.input_feature

        new_bases_comp = tf.nn.softmax(tf.Variable(np.zeros((FLAGS.n_new_bases, 3)), trainable = True, dtype="float32"), axis = 1)
        self.bases_comp = tf.concat([tf.constant(np.load(FLAGS.bases_comp_dir)[:FLAGS.n_bases], dtype = "float32", name = "bases_comp"), new_bases_comp], axis = 0)


        raw_comp_prime = tf.matmul(self.rescaled_activation, self.bases_comp)

        self.comp_prime = sum_normalize(raw_comp_prime)

        #tf.transpose(tf.transpose(raw_comp_prime) / tf.reduce_sum(raw_comp_prime, axis = 1))


        #self.comp_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.comp - self.comp_prime), axis = 1))
        self.comp_loss_batch = KL_dis(self.comp_prime, self.comp, FLAGS.eps, 1.0)
        self.comp_loss = tf.reduce_mean(self.comp_loss_batch)
        if (is_training):
            tf.summary.scalar('train/comp_loss * %.6f'%FLAGS.comp_decay, self.comp_loss)
        
        cond = tf.cast(tf.logical_and(tf.equal(tf.reduce_mean(self.degree_of_freedom), 2.0), \
        tf.greater(tf.reduce_sum(self.input_feature * tf.constant([0.0, 0.0, 1.0], dtype = "float32")), 1e-6)), tf.float32)
        fac = 1.0 * (1 - cond) + 100.0 * cond
        tf.summary.scalar('train/fac ', fac)
        
        self.JS_dis_batch = JS_dis(self.normalized_input_xrd, self.xrd_prime, fac)
        self.L2_dis_batch = L2_dis(self.normalized_input_xrd, self.xrd_prime)
        self.L1_dis_batch = L1_dis(self.normalized_input_xrd, self.xrd_prime)
        self.recon_loss_batch = self.JS_dis_batch + self.L2_dis_batch + self.L1_dis_batch
        self.recon_loss = tf.reduce_mean(self.recon_loss_batch)
        self.sqr_recon_loss = tf.reduce_mean(tf.square(self.recon_loss_batch))

        if (is_training):
            tf.summary.scalar('train/recon_loss', self.recon_loss)
            tf.summary.scalar('train/JS_dis', tf.reduce_mean(self.JS_dis_batch))
            tf.summary.scalar('train/L2_dis', tf.reduce_mean(self.L2_dis_batch))
            tf.summary.scalar('train/L1_dis', tf.reduce_mean(self.L1_dis_batch))
            tf.summary.scalar('train/sqr_recon_loss', self.sqr_recon_loss)
    
        self.vae_loss = self.recon_loss #+ self.tot_kl_loss * FLAGS.beta
        if (is_training):
            tf.summary.scalar('train/vae_loss', self.vae_loss)

        ######  gibbs-alloy loss  #################
        act = self.weights 
        P = tf.pow(self.weights + FLAGS.eps, FLAGS.beta)
        P = tf.transpose(tf.transpose(P) / (tf.reduce_sum(P, axis = 1))) #bs, n_bases
        mean_loss = FLAGS.mean_loss #self.recon_loss
        self.gibbs_loss_batch = tf.reduce_sum(- P * tf.log(P + FLAGS.eps), axis = 1) # bs 


        self.prev_shift = tf.slice(self.mu_shift, [0, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.n_bases + FLAGS.n_new_bases])
        self.next_shift = tf.slice(self.mu_shift, [1, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.n_bases + FLAGS.n_new_bases])

        self.prev_act = tf.slice(act, [0, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.n_bases + FLAGS.n_new_bases])
        self.next_act = tf.slice(act, [1, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.n_bases + FLAGS.n_new_bases])

        delta = - 0.05 * FLAGS.shift_unit 
        if ("refine" in sys.argv):
            delta = 0.05 * FLAGS.shift_unit
            print("refine mode")
        
        share_act_bases = tf.stop_gradient(tf.cast(tf.greater(tf.minimum(self.prev_act, self.next_act), FLAGS.min_activation), tf.float32)) #bs-1, n_bases

        shift_between = tf.reduce_max(tf.abs(self.prev_shift - self.next_shift) * share_act_bases, axis = 1)
         
        max_shift_between = tf.reduce_max(tf.abs(self.prev_shift - self.next_shift) * tf.cast(tf.greater(tf.minimum(self.prev_act, self.next_act), FLAGS.min_activation), tf.float32), axis = 1) #bs-1
        self.max_shift = tf.concat([max_shift_between, tf.constant([0.0])], axis = 0)


        shift_penalty = tf.nn.leaky_relu(tf.tanh((shift_between - (FLAGS.shift_unit + delta)) / FLAGS.shift_amplify), alpha = 0.1)


        diff = tf.maximum(tf.concat([shift_penalty, tf.constant([0.0])], axis = 0), tf.concat([tf.constant([0.0]), shift_penalty], axis = 0)) #d_1,2, d_1,2 + d2,3, ..., d_8,9 + d_9,10, d_9,10: penalties
        #n_diff = tf.stop_gradient(tf.concat([shift_penalty*0.0 + 1, tf.constant([0.0])], axis = 0) + tf.concat([tf.constant([0.0]), shift_penalty*0.0 + 1], axis = 0)) # 1,2,...,2,1
        avg_diff = diff #/ n_diff #bs

        self.alloy_loss_batch = avg_diff * (tf.log(self.degree_of_freedom) - tf.log(self.degree_of_freedom - 1 + FLAGS.eps))

        tf.summary.histogram("gibbs_loss", tf.reshape(self.gibbs_loss_batch, [-1]))
        tf.summary.histogram("alloy_loss", tf.reshape(self.alloy_loss_batch, [-1]))

        self.condition1 = tf.greater(tf.reduce_sum(tf.cast(tf.greater(act, FLAGS.min_activation), tf.float32), axis = 1), 0.5 + self.degree_of_freedom) #bs n_bases > d_free
        self.condition2 = tf.greater(tf.reduce_sum(tf.cast(tf.greater(act, FLAGS.min_activation), tf.float32), axis = 1), -0.5 + self.degree_of_freedom) #bs n_bases >= d_free

        is_violated = tf.cast(tf.logical_and(tf.greater(tf.abs(self.prev_shift - self.next_shift), FLAGS.shift_unit + delta), tf.greater(tf.minimum(self.prev_act, self.next_act), FLAGS.min_activation)), tf.float32) #bs - 1
        is_violated = tf.reduce_sum(is_violated, axis = 1) 
        num_violate = tf.concat([is_violated, tf.constant([0.0])], axis = 0) + tf.concat([tf.constant([0.0]), is_violated], axis = 0)

        #condition3 = tf.greater_equal(self.alloy_loss_batch, 0.04) #bs shift = true
        self.condition3 = tf.greater(num_violate, 0.5) #bs shift = true
        condition = tf.logical_or(self.condition1, tf.logical_and(self.condition2, self.condition3))
        
        penalty_ratio = tf.maximum(tf.stop_gradient(tf.cast(condition, tf.float32)), FLAGS.gibbs_penalty)
        self.gibbs_penalty_ratio = penalty_ratio
        
        #if (is_training):
        #    tf.summary.scalar('train/percent_of_unsatisfied', tf.reduce_mean(tf.cast(condition, tf.float32)))
        
        #only decrease the entropy over some threshold
        #tf.stop_gradient(tf.nn.sigmoid((mean_loss - self.recon_loss_batch))) # bs  
        
        self.gibbs_loss = tf.reduce_mean((self.gibbs_loss_batch + FLAGS.alloy_decay * self.alloy_loss_batch) * penalty_ratio) # / (FLAGS.eps + tf.reduce_sum(penalty_ratio))
        self.gibbs_decay = 0.0 + tf.minimum(self.epoch, 10.0) * FLAGS.gibbs_decay / 10.0


        if (is_training):
            tf.summary.scalar('train/gibbs_loss', self.gibbs_loss)
            tf.summary.scalar('train/gibbs_decay', self.gibbs_decay)


        ##### smooth_weights_loss ######
        


        self.prev_weights = tf.slice(self.weights, [0, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.n_bases + FLAGS.n_new_bases])
        self.next_weights = tf.slice(self.weights, [1, 0], [tf.shape(self.input_feature)[0] - 1, FLAGS.n_bases + FLAGS.n_new_bases])
        
        almost_zero = 1.0 #tf.stop_gradient(tf.cast(tf.logical_or(tf.greater(0.05, self.prev_weights), tf.greater(0.05, self.next_weights)), tf.float32))
        
       
       
        self.smooth_weights_loss_batch = tf.reduce_sum(tf.maximum(tf.abs(self.prev_weights - self.next_weights) - FLAGS.smooth_weights_th, 0.0), axis = 1) #/ self.distance * tf.reduce_min(self.distance)
        #tf.sqrt(tf.reduce_sum(tf.square(self.prev_weights - self.next_weights) * almost_zero, axis = 1) + FLAGS.eps)
        #tf.norm(self.prev_weights - self.next_weights, axis=1) #/ self.distance

        self.smooth_weights_loss = tf.reduce_mean(self.smooth_weights_loss_batch)

        self.smoothness_decay = FLAGS.smoothness_decay
        #self.smoothness_decay = tf.minimum(self.epoch, 10.0) * FLAGS.smoothness_decay / 10.0

        if (is_training):
            tf.summary.scalar('train/smooth_weights_loss * %.6f'%FLAGS.smoothness_decay, self.smooth_weights_loss)


        ####### l2 loss ##########

        self.l2_loss = tf.add_n(tf.losses.get_regularization_losses()) #+FLAGS.weight_decay*tf.nn.l2_loss(self.r_sqrt_sigma)
        if (is_training):
            tf.summary.scalar('train/l2_loss', self.l2_loss)


        ####### total loss ##########
        if (is_training):
            self.total_loss = self.l2_loss + self.vae_loss \
                          + self.gibbs_loss * self.gibbs_decay \
                          + self.smooth_weights_loss * self.smoothness_decay\
                          + self.comp_loss * FLAGS.comp_decay \
                          + self.sqr_recon_loss * FLAGS.sqr_recon_loss_decay\
                          + self.intensity_shift_loss * FLAGS.intensity_shift_loss_decay 
        else: 
            self.total_loss = self.vae_loss

        if (is_training):
            tf.summary.scalar('train/total_loss', self.total_loss)

        self.optimizer_loss = self.total_loss

