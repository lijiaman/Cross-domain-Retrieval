import tensorflow as tf
import numpy as np
import nin
import utils
import input
import time

ISOTIMEFORMAT='%Y-%m-%d %X'
tmp = '/ais/gobi4/fashion/data/Category-Attribute-Prediction/'
with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    images = tf.placeholder(tf.float32, [None, 227, 227, 3])
    true_out = tf.placeholder(tf.float32, [None, 50])
    train_mode = tf.placeholder(tf.bool)

    nin = nin.NIN()
    nin.build(images, train_mode)

    # print number of variables used
    print nin.get_var_count()

    #sess.run(tf.initialize_all_variables())

    # test classification
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(nin.final,true_out))
    #loss = tf.reduce_mean(-tf.reduce_sum(true_out*tf.log(nin.prob),[1]))
    #loss = tf.reduce_mean(tf.reduce_sum((nin.prob-true_out)**2,[1]))
    #train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    global_step = tf.Variable(0)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
    sess.run(tf.initialize_all_variables())   
    for i in xrange(10000):
    	x_batch, y_batch = input.load_batchsize_images(batch_size=64)
    	#cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    	batch_size = y_batch.shape[0]
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(nin.prob,true_out))
        #print loss.eval(feed_dict={images: x_batch, true_out: y_batch, train_mode: True})
        #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
	lr,prob,f, pool,cost, _ = sess.run([learning_rate,nin.prob,nin.final,nin.pool4,loss,optimizer], feed_dict={images: x_batch, true_out: y_batch, train_mode: True})
        #print "np.argsort:"
        #print np.argsort(prob,axis=1).shape
        index = np.argsort(prob,axis=1)[:,49:50]
        index_3 = np.argsort(prob,axis=1)[:,47:50]
        index_5 = np.argsort(prob,axis=1)[:,45:50]
        #print "x_batch:"
        #print x_batch
        #print "pool4"
        #print pool
        #time.sleep(5)
        #print "prob:"
        #print prob
        #print "final"
        #print f
        #time.sleep(3)
       # print "index:"
        #print index
        #time.sleep(3)
        #print index
        #print "y-batch[index]:"
        #print y_batch[index]
        #print "y_batch[index]==1:"
        #print y_batch[index] == 1
        #print "y_batch:"
        #print y_batch
        #print y_batch[xrange(batch_size),index.reshape(-1)]
        acc = np.mean(y_batch[xrange(batch_size),index.reshape(-1)].reshape(batch_size,-1) == 1)
        if i % 10 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print print_time
            print('Train iteration:{0}, loss:{1}, accuracy:{2}, lr:{3}'.format(i, cost, acc, lr))
        if i % 100 == 0:
            x_batch_test, y_batch_test = input.load_test_images(test_num=64)
            prob_test = sess.run(nin.prob, feed_dict={images: x_batch_test, train_mode: False})
            test_num = y_batch_test.shape[0]
            index_test = np.argsort(prob_test,axis=1)[:,49:50]

            acc = np.mean(y_batch_test[xrange(test_num),index_test.reshape(-1)].reshape(test_num,-1) == 1)
            print('Test After iterations:{0}, Test Accuracy:{1}'.format(i, acc))
        
    #for i in xrange(1000):
    	#x_batch_test, y_batch_test = input.load_test_images(test_num=500)
    	#prob = sess.run(nin.prob, feed_dict={images: x_batch_test, train_mode: False})
    	#utils.print_prob(prob[0], tmp+'synset.txt')
        #print "prob[0]:"
        #print prob[0]
        #test_num = y_batch_test.shape[0]
        #index = np.argsort(prob,axis=1)[:,49:50]
        #acc = np.mean(y_batch_test[xrange(test_num),index.reshape(-1)].reshape(test_num, -1) == 1)
        #print('Test iteration:{0}, Test Accuracy:{1}'.format(i, acc))
    # test save
   # nin.save_npy(sess, tmp+'test-save.npy')
