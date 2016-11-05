import random
import sys,os
import numpy as np
import subprocess
import time
import cPickle
import tensorflow as tf
#from tensorflow.python.ops import RNNCell
#from tensorflow.python.ops import variable_scope as vs

def bound(f_in,f_out):
    return np.sqrt(6./(f_in+f_out))

def smoothing(xl):
    sig_x = tf.nn.sigmoid(xl)
    return sig_x/tf.to_float(tf.reduce_sum(sig_x,1))

def cosine(M, x):
    # M(bs,de)*x(de,) = c(bs,1)
    episilon = tf.constant(1e-12)
    M_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.mul(M,M),1),episilon))
    x_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.mul(x,x)),episilon))
    prod = tf.reduce_sum(tf.mul(M,x),1)
    return ((prod/(M_norm*x_norm))+1.)/2.

def prod_sim(a,B,U):
    ## aUUb
    new_a = tf.expand_dims(a,0)
    return tf.squeeze(tf.matmul(tf.matmul(new_a,U),tf.matmul(U,B,transpose_a=True,transpose_b=True)))

def tensor_prod(x,y,w_l,w_r):
    w_shape = tf.shape(w_l)     # (hidden,rank,feature)
    x = tf.transpose(x)
    y = tf.transpose(y)
    w2_l = tf.reshape(w_l,[-1,w_shape[-1]])
    w2_r = tf.reshape(w_r,[-1,w_shape[-1]])
    h_l = tf.matmul(w2_l,x)     # (hidden*rank,feature)*(feature,1) = (hidden*rank,1)
    h_r = tf.matmul(w2_r,y)
    prod = tf.reshape(h_l*h_r,[w_shape[0],w_shape[1]])      # (hidden,rank)
    return tf.sigmoid(tf.reduce_sum(prod,1))		# (hidden,1)

class Model:
    def __init__(self,vec,sl,de,nc,ps,lr,bs,nh,nf,nm=400,rank=2):
        with tf.device("/gpu:0"):
            self.features = tf.placeholder(tf.float32,shape=(bs,nf),name='features')
            self.idxs = tf.placeholder(tf.int32,shape=(bs,sl),name='indexes')
            x = tf.nn.embedding_lookup(vec,self.idxs,name='x')
            self.y = tf.placeholder(tf.int32,shape=(bs,),name='y')
            self.word_mask = tf.placeholder(tf.float32,shape=(bs,sl),name='word_mask')
            new_mask = tf.expand_dims(self.word_mask,-1,name='new_mask')
        with tf.device("/cpu:0"):
            self.sen_mask = tf.placeholder(tf.bool,shape=(bs,),name='sen_mask')
            w2 = tf.Variable(tf.random_uniform(minval=-bound(nf,10*nf),maxval=bound(nf,10*nf),shape=(nf,10*nf),name='w2'))
            b2 = tf.Variable(tf.zeros([nf*10]),name='b2')
            w1 = tf.Variable(tf.random_uniform(shape=(nh,nc),minval=-bound(nh,nc),maxval=bound(nh,nc)),name='weights')
            b1 = tf.Variable(tf.zeros([nc]),name='biases')
            # add in the memory module
            U = tf.Variable(tf.random_uniform(shape=(nh,nm),minval=-bound(nh,nm),maxval=bound(nh,nm)),name='match_weights')
            wg_1 = tf.Variable(tf.random_uniform(shape=(nh,nh),minval=-bound(nh,nh),maxval=bound(nh,nh)),name='wg_1')
            bg = tf.Variable(tf.zeros([nh]),name='bg')
            wg_2 = tf.Variable(tf.random_uniform(shape=(nh,nh),minval=-bound(nh,nh),maxval=bound(nh,nh)),name='wg_2')
            wg_3 = tf.Variable(tf.random_uniform(shape=(nh,nh),minval=-bound(nh,nh),maxval=bound(nh,nh)),name='wg_3')
            wl = tf.Variable(tf.random_uniform(shape=(nh,rank,nh),minval=-bound(nh,nh),maxval=bound(nh,nh)),name='wl')
            wr = tf.Variable(tf.random_uniform(shape=(nh,rank,nh),minval=-bound(nh,nh),maxval=bound(nh,nh)),name='wr')
            

            # the following module is for the intra-sentence level embedding, for now the memory doesn't play any roles
            # however, in the future, attention mechanism may come to help
        with tf.device("/gpu:0"):
            with tf.variable_scope("utt"):
                utt_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=nh)
                utt_outputs = []
                utt_last_state = utt_state = utt_init_state = utt_rnn_cell.zero_state(batch_size=bs,dtype=tf.float32)
                for time_step in xrange(sl):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    utt_output,utt_state = utt_rnn_cell(x[:,time_step,:],utt_last_state)
                    utt_last_state = utt_last_state*(tf.ones_like(new_mask[:,time_step,:])-new_mask[:,time_step,:])+utt_state*new_mask[:,time_step,:]
                    utt_outputs.append(utt_output*new_mask[:,time_step,:])
                utt_outputs = tf.transpose(tf.pack(utt_outputs),[1,0,2])
                sen_vec = tf.reduce_max(utt_outputs,1)


            feat_out = tf.nn.relu(tf.matmul(self.features,w2)+b2)
            combine_output = tf.concat(concat_dim=1,values=[sen_vec,feat_out])

            # the following part is for the inter-sentence level
        with tf.device("/cpu:0"):
            with tf.variable_scope("conv"):
                conv_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=nh)
                conv_outputs = []
                conv_state = conv_init_state = conv_rnn_cell.zero_state(batch_size=1,dtype=tf.float32)
                for time_step in xrange(bs):
                    max_score = -1
                    if time_step > 0:tf.get_variable_scope().reuse_variables()
                        # first, we would like to combine the previous memory with the current sentence, then put the new memory into the memory block with a gate
                        # output_module should be added here to find the first supporting fact
                    o1_index = 0
                    memory = tf.concat(concat_dim=0,values=[sen_vec[:time_step,:],tf.zeros([bs-time_step,nh])])
                    sim_score = prod_sim(sen_vec[time_step,:],memory,U)
                    #sim_score = cosine(memory,sen_vec[time_step,:])
                    o1_index = tf.to_int32(tf.argmax(sim_score, dimension=0,name='max_similarity'))
                    o1 = tf.expand_dims(memory[o1_index,:],0)
                    # there should be a gate to controll the flow of sentence vector into the memory
                    # (1,nh)*(nh,nh) = (1,nh)
                    G = tf.sigmoid(tf.matmul(conv_state,wg_1)+tf.matmul(tf.expand_dims(sen_vec[time_step,:],0),wg_2)+tf.matmul(o1,wg_3)+bg)
                    o1 = G*o1
                    # here, we need to consider how to merge the o1 into the gru cell
                    combine = tf.reshape(tf.concat(concat_dim=1,values=[tf.expand_dims(sen_vec[time_step,:],0),o1]),[-1,2*nh])         # (1,640)
                    #combine = tf.reshape(tensor_prod(tf.expand_dims(sen_vec[time_step,:],0),o1,wl,wr),[-1,nh])
                    #conv_output,conv_state = conv_rnn_cell(tf.expand_dims(sen_vec[time_step,:],0),conv_state)
                    conv_output,conv_state = conv_rnn_cell(combine,conv_state)
                    sm = tf.matmul(conv_output,w1)+b1
                    conv_outputs.append(sm)
                conv_outputs = tf.concat(concat_dim=0,values=conv_outputs)
            

        with tf.device("/cpu:0"):
            self.nll = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.boolean_mask(conv_outputs,self.sen_mask),tf.boolean_mask(tf.to_int64(self.y),self.sen_mask)))
            optimizer = tf.train.AdamOptimizer(lr)
            self.train = optimizer.minimize(self.nll)
            right = tf.boolean_mask(tf.equal(self.y,tf.cast(tf.argmax(conv_outputs,1),tf.int32)),self.sen_mask)
            self.right_num = tf.reduce_sum(tf.cast(right,tf.float32))
            self.tags = tf.boolean_mask(tf.to_int32(tf.argmax(conv_outputs,1)),self.sen_mask)

            ## debugger methods
            self.mem_index = combine

    print 'model completed'

