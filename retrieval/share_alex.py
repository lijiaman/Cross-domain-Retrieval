import tensorflow as tf
import os
import numpy as np
import time
import inspect
import skimage
import skimage.io
import skimage.transform
import input
import alex



sess = tf.InteractiveSession()
#batch_size = 64
batch_size = 24
#batch_size = 1
x_street = tf.placeholder("float", [batch_size, 227, 227, 3])
x_shop = tf.placeholder("float", [batch_size, 227, 227, 3])
if_pair = tf.placeholder("float", [batch_size,])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
#street_network = alex.ALEXNET(alex_npy_path=None, trainable=True)
street_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=True)
#shop_network = alex.ALEXNET(alex_npy_path=None, trainable=True)
shop_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=True)

street_network.build(rgb=x_street, flag="share", train_mode=train_mode)
shop_network.build(rgb=x_shop, flag="share", reuse=True, train_mode=train_mode)

y_street = street_network.relu6
y_shop = shop_network.relu6

dist_square_vec = tf.reduce_sum(tf.square(tf.sub(y_street, y_shop)), 1)

ones = tf.constant(1.0, dtype="float", shape=[batch_size,])
zero = tf.constant(0.0, dtype="float", shape=[batch_size,])
margin = tf.constant(0.03, dtype="float", shape=[batch_size,])
#For pair
#pred_pair = tf.less(dist_square_vec, margin)
#triplet_loss_pair = tf.select(pred_pair, dist_square_vec, margin)
triplet_loss_pair = tf.minimum(margin, dist_square_vec)
#For not pair
dist = tf.sub(margin, dist_square_vec)
#pred_nopair = tf.less(dist, zero)
#triplet_loss_nopair = tf.select(pred_nopair, zero, dist)
triplet_loss_nopair = tf.maximum(zero, dist)

dist_square = tf.reduce_mean(dist_square_vec)
#pred = tf.less(zero, if_pair)
#triplet_loss_vec = tf.select(pred, triplet_loss_pair, triplet_loss_nopair)
triplet_loss_vec = tf.add(tf.mul(if_pair, triplet_loss_pair), tf.mul(tf.sub(ones, if_pair), triplet_loss_nopair))
triplet_loss = tf.reduce_mean(triplet_loss_vec)
tf.scalar_summary('Loss', triplet_loss)

merged = tf.merge_all_summaries()
log_dir = 'alex_triplet_share'
train_writer = tf.train.SummaryWriter(log_dir+'/train')
val_writer = tf.train.SummaryWriter(log_dir+'/val')
# train
n_epoch = 5000
global_step = tf.Variable(0)
starter_learning_rate = 0.000001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 15, 0.96, staircase=True)
print_freq = 1

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(triplet_loss, global_step=global_step)
#train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)

sess.run(tf.initialize_all_variables())


ISOTIMEFORMAT = '%Y-%m-%d %X'
iter_show = 0
for epoch in range(n_epoch):
    start_time = time.time()
    #iter_per_epoch = 3068
    iter_per_epoch =1
    for iter in xrange(iter_per_epoch):
    	street_batch, shop_batch, y_street_batch, y_shop_batch, if_pair_batch = input.load_share_train(batch_size)
        feed_dict = {x_street: street_batch, x_shop: shop_batch, if_pair: if_pair_batch, train_mode:True}
        _, err, d_s, lr, train_summary = sess.run([train_op, triplet_loss, dist_square, learning_rate, merged], feed_dict=feed_dict)
        iter_show += 1
        train_writer.add_summary(train_summary, iter_show)

        if epoch % 10 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Train Loss: %f" % err)
            print("   Dist_square: %f" % d_s)
            #print("pred_out:")
            #print(pred_out.shape)            

 
#    val_triplet_loss, val_dsquare, batch_val_size = 0, 0, 32
#
#    #val_iters = 1518
#    val_iters = 2
#      
#    for iter_val in xrange(val_iters):
#        street_val_batch, shop_val_batch, y_street_val_batch, y_shop_val_batch, if_val_batch = input.load_share_val(batch_val_size)
#        
#        feed_dict = {x_street: street_val_batch, x_shop: shop_val_batch, if_pair: if_val_batch, train_mode:False}
#        
#        val_err, d_val_square = sess.run([triplet_loss, dist_square], feed_dict=feed_dict)
#        val_triplet_loss += val_err
#        val_dsquare += d_val_square
#
##
#    loss_val_f = val_triplet_loss / val_iters
#    d_square_val_f = val_dsquare / val_iters
#    print("   val triplet loss: %f" % loss_val_f)
#    print("   dist_share: %f" % d_square_val_f)
#    
#    summary = tf.Summary()
#    summary.value.add(tag="Loss",simple_value=loss_val_f)
#    val_writer.add_summary(summary, iter_show)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
#        
#        test_triplet_loss, test_dsquare, batch_test_size = 0, 0, 32
#        #test_iters = 1481
#        test_iters = 2
#        #test_iters = 1480
#        for iter_test in xrange(test_iters):
#            street_test_batch, shop_test_batch, y_street_test_batch, y_shop_test_batch, if_test_batch = input.load_share_test(batch_test_size)
#            
#            feed_dict = {x_street: street_test_batch, x_shop: shop_test_batch, if_pair: if_test_batch, train_mode:False}
#            
#            test_err, d_test_square = sess.run([triplet_loss, dist_square], feed_dict=feed_dict)
#            test_triplet_loss += test_err
#            test_dsquare += d_test_square
#
##
#        loss_out = test_triplet_loss / test_iters
#        d_out = test_dsquare / test_iters
#        print("   test triplet loss: %f" % loss_out)
#        print("   dist_square: %f" % d_out)
#        with open('/ais/gobi4/fashion/retrieval/share_total_test_result.txt', 'a') as w_f:
#            w_f.write(str(epoch+1)+'\t'+str(loss_out)+'\t'+str(d_out)+'\n')
#        w_f.close()

        street_network.save_npy(sess=sess,npy_path="/ais/gobi4/fashion/retrieval/share_street_alex.npy")
        shop_network.save_npy(sess=sess,npy_path="/ais/gobi4/fashion/retrieval/share_shop_alex.npy")

train_writer.close()
val_writer.close()
