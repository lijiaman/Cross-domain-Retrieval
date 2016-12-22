import tensorflow as tf
import os
import numpy as np
import time
import inspect
import skimage
import skimage.io
import skimage.transform
import input
import native_alex



sess = tf.InteractiveSession()
batch_size = 128
x = tf.placeholder("float", [None, 227, 227, 3])
y_ = tf.placeholder(tf.int32, shape=[None,])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
network = native_alex.ALEXNET(alex_npy_path=npy_path, trainable=True)
network.build(x, train_mode)
y = network.fc8
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, y_, 5, 'top-5'), tf.float32))
top10_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, y_, 10, 'top-10'), tf.float32))
tf.scalar_summary('Loss', cost)
tf.scalar_summary('Top1-Acc', acc)
tf.scalar_summary('Top5-Acc', top5_acc)
tf.scalar_summary('Top10-Acc', top10_acc)

merged = tf.merge_all_summaries()
log_dir = 'alex_full'
train_writer = tf.train.SummaryWriter(log_dir+'/train')
val_writer = tf.train.SummaryWriter(log_dir+'/val')
# train
n_epoch = 30
global_step = tf.Variable(0)
starter_learning_rate = 0.000001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
print_freq = 1

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, global_step=global_step)
#train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)

sess.run(tf.initialize_all_variables())


ISOTIMEFORMAT = '%Y-%m-%d %X'
iter_show = 0
for epoch in range(n_epoch):
    start_time = time.time()
    iter_per_epoch = 1635
    for iter in xrange(iter_per_epoch):
    	x_batch, y_batch = input.load_batchsize_images(batch_size)
        feed_dict = {x: x_batch, y_: y_batch, train_mode:True}
        #conv1, conv2, conv3, conv4, conv5, fc8, fc7, fc6, pool3 = sess.run([network.conv1, network.conv2, network.conv3, network.conv4, network.conv5, network.fc8, network.fc7, network.fc6, network.pool3], feed_dict=feed_dict)
        _, err, ac, top5_ac, top10_ac, lr, train_summary = sess.run([train_op, cost, acc, top5_acc, top10_acc, learning_rate, merged], feed_dict=feed_dict)
        iter_show += 1
        train_writer.add_summary(train_summary, iter_show)
        #network.save_npy(sess=sess, npy_path="test_save.npy")

        if iter % 10 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Train Loss: %f" % err)
            print("   Top1 Acc: %f" % ac)
            print("   Top5 Acc: %f" % top5_ac)
            print("   Top10 Acc: %f" % top10_ac)

        x_val_batch, y_val_batch = input.load_val_images(batch_size)
        feed_dict_val = {x: x_val_batch, y_: y_val_batch, train_mode:False}
        val_err, val_ac, val_top5_ac, val_top10_ac, val_summary = sess.run([cost, acc, top5_acc, top10_acc, merged], feed_dict=feed_dict_val)
        val_writer.add_summary(val_summary, iter_show)  
        if iter % 10 == 0:      
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Val Loss: %f" % val_err)
            print("   Val Top1 Acc: %f" % val_ac)
            print("   Val Top5 Acc: %f" % val_top5_ac)
            print("   Val Top10 Acc: %f" % val_top10_ac)


    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        test_loss, test_acc, test_top5_acc, test_top10_acc, batch_test_size = 0, 0, 0, 0, 128
        test_iters = 312
        for iter_test in xrange(test_iters):
            #print iter_test
            x_test_batch, y_test_batch = input.load_test_images(batch_test_size)
            feed_dict = {x: x_test_batch, y_: y_test_batch, train_mode:False}
            err, ac, top5_ac, top10_ac = sess.run([cost, acc, top5_acc, top10_acc], feed_dict=feed_dict)
            test_loss += err; test_acc += ac; test_top5_acc += top5_ac; test_top10_acc += top10_ac
        print("   Test Loss: %f" % (test_loss/ test_iters))
        print("   Test Top1-Acc: %f" % (test_acc/ test_iters))
        print("   Test Top5-Acc: %f" % (test_top5_acc / test_iters))
        print("   Test Top10-Acc: %f" % (test_top10_acc / test_iters))
        with open('test_result.txt','a') as w_f:
            w_f.write(str(epoch)+'\t'+str(test_loss/test_iters)+'\t'+str(test_acc/test_iters)+'\t'+str(test_top5_acc/test_iters)+'\t'+str(test_top10_acc/test_iters)+'\n')
        w_f.close()
        network.save_npy(sess=sess)

train_writer.close()
val_writer.close()
