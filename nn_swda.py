import numpy as np
import cPickle, gzip, numpy
import sys
import time
import tensorflow as tf
from memNN_model import Model
import random

"""
there are two parameters for this program, 
first one is training steps
second one is learning rate
"""
def bound(f_in,f_out):
    return np.sqrt(6./(f_in+f_out))

def get_mask(xl):
    return np.array([np.array([num > 0 for num in x],dtype='float32') for x in xl])

def get_senlen(xl):
    return np.sum(np.array([np.array([num > 0 for num in x],dtype='float32') for x in xl]),axis=1)

s = {'nhidden':300,
    'nc':44,
    'lr':0.001,
    'de':300,
    'sl':127,
    'sn':542,
    'nh':300,
    'nepochs':10,
    'bs':542,
    'sample_size':1,
    'nf':4,
    }
# data set
train_data,test_data,valid_data,vec = cPickle.load(open('swda.pkl','rb'))
vec_size = np.array(vec).shape
print 'vector size',vec_size
vec = tf.to_float(tf.Variable(np.array(vec).astype('float32'),name='vec',trainable=True))

# parameters
poolout_size = 300    # the size of a sentence after pooling
early_stop = True

cnn = Model(vec,s['sl'],s['de'],s['nc'],poolout_size,s['lr'],s['bs'],s['nh'],s['nf'])

# training
best_acc = 0
best_epoch = 0
patience = 0
saver = tf.train.Saver()
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(s['nepochs']):
            right_total = 0
            total_err = 0
            tic = time.time()
            random.shuffle(train_data)
            for j in range(len(train_data)/s['sample_size']):
                text,labels,b_mask,s_mask,features = train_data[j]
                labels = np.argmax(labels,1)
                _,err,o1_index = sess.run([cnn.train,cnn.nll,cnn.mem_index],feed_dict={cnn.idxs:text,cnn.y:labels,cnn.sen_mask:s_mask, cnn.word_mask:b_mask,cnn.features:np.array(features,dtype='float32')})
                print '[learning] epoch %i >> %2.2f%% nll %f'%(i,(j+1)*100./len(train_data)*s['sample_size'],err),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()
            right_total = 0
            total_label = 0
            for j in range(len(valid_data)):
                text,labels,b_mask,s_mask,features = valid_data[j]
                labels = np.argmax(labels,1)
                r_num = sess.run(cnn.right_num,feed_dict={cnn.idxs:text,cnn.y:labels,cnn.sen_mask:s_mask, cnn.word_mask:b_mask,cnn.features:np.array(features,dtype='float32')})
                right_total += r_num
                total_label += np.sum(s_mask)
            acc = right_total/float(total_label)
            print 'validation accuracy',acc
            sys.stdout.flush()
            if acc > best_acc:
                best_epoch = i
                save_path = saver.save(sess,"./memNN_model.ckpt")
                best_acc = acc
                patience = 0
            else:
                patience += 1
            if early_stop and patience > 5:
                break

        saver.restore(sess,"./memNN_model.ckpt")
        right_total = 0
        total_label = 0
        for j in range(len(test_data)):
            text,labels,b_mask,s_mask,features = test_data[j]
            labels = np.argmax(labels,1)
            r_num = sess.run(cnn.right_num,feed_dict={cnn.idxs:text,cnn.y:labels,cnn.sen_mask:s_mask, cnn.word_mask:b_mask,cnn.features:np.array(features,dtype='float32')})
            right_total += r_num
            total_label += np.sum(s_mask)
        acc = right_total/float(total_label)
        print 'test accuracy',acc
        sys.stdout.flush()
        print 'best validation accuracy',best_acc,'at epoch',best_epoch
