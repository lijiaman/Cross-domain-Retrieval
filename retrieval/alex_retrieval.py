import tensorflow as tf
import tensorlayer as tl
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
batch_size= 64
#batch_size = 32
x_street = tf.placeholder("float", [batch_size, 227, 227, 3])
x_shop = tf.placeholder("float", [batch_size, 227, 227, 3])
x_nopair = tf.placeholder("float", [batch_size, 227, 227, 3])
#y_ = tf.placeholder(tf.int32, shape=[None,])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
street_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=True)
shop_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=True)
nopair_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=True)

street_network.build(rgb=x_street, flag="street", train_mode=train_mode)
shop_network.build(rgb=x_shop, flag="shop", train_mode=train_mode)
nopair_network.build(rgb=x_nopair, flag="shop", reuse=True, train_mode=train_mode)

y_street = street_network.relu6
y_shop = shop_network.relu6
y_nopair = nopair_network.relu6

dist_pair_vec = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(y_street, y_shop)), 1))
dist_nopair_vec = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(y_street, y_nopair)), 1))
dist_pair = tf.reduce_mean(dist_pair_vec)
dist_nopair = tf.reduce_mean(dist_nopair_vec)


zero = tf.constant(0.0, dtype="float", shape=[batch_size,])
margin = tf.constant(0.3, dtype="float", shape=[batch_size,])
dist_vec = tf.add(tf.sub(dist_pair_vec, dist_nopair_vec), margin)
pred = tf.less(dist_vec, zero)
triplet_loss_vec = tf.select(pred, zero, dist_vec)
triplet_loss = tf.reduce_mean(triplet_loss_vec)
tf.scalar_summary('Loss', triplet_loss)

merged = tf.merge_all_summaries()
log_dir = 'alex_triplet_noshare'
train_writer = tf.train.SummaryWriter(log_dir+'/train')
val_writer = tf.train.SummaryWriter(log_dir+'/val')
# train
n_epoch = 3000
global_step = tf.Variable(0)
starter_learning_rate = 0.00001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 6000, 0.96, staircase=True)
print_freq = 1

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(triplet_loss, global_step=global_step)
#train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)

sess.run(tf.initialize_all_variables())


ISOTIMEFORMAT = '%Y-%m-%d %X'
iter_show = 0
for epoch in range(n_epoch):
    start_time = time.time()
    iter_per_epoch = 1534
    #iter_per_epoch = 2
    for iter in xrange(iter_per_epoch):
    	street_batch, shop_batch, nopair_batch, _, _, _ = input.load_batchsize_images(batch_size)
        feed_dict = {x_street: street_batch, x_shop: shop_batch, x_nopair: nopair_batch, train_mode:True}
        #conv1, conv2, conv3, conv4, conv5, fc8, fc7, fc6, pool3 = sess.run([network.conv1, network.conv2, network.conv3, network.conv4, network.conv5, network.fc8, network.fc7, network.fc6, network.pool3], feed_dict=feed_dict)
        _, err, d_pair, d_nopair, lr, train_summary = sess.run([train_op, triplet_loss, dist_pair, dist_nopair, learning_rate, merged], feed_dict=feed_dict)
        iter_show += 1
        train_writer.add_summary(train_summary, iter_show)
        #network.save_npy(sess=sess, npy_path="test_save.npy")

        if iter % 10 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Train Loss: %f" % err)
            print("   Dist pair: %f" % d_pair)
            print("   Dist no pair: %f" % d_nopair)

    val_triplet_loss, val_dpair, val_nopair, batch_val_size = 0, 0, 0, 64
    val_iters = 759
    #val_iters = 2
      
    for iter_val in xrange(val_iters):
        street_val_batch, shop_val_batch, nopair_val_batch, y_street_val_batch, y_shop_val_batch, y_nopair_val_batch = input.load_val_images(batch_val_size)
        
        feed_dict = {x_street: street_val_batch, x_shop: shop_val_batch, x_nopair: nopair_val_batch, train_mode:False}
        
        val_err, d_val_pair, d_val_nopair = sess.run([triplet_loss, dist_pair, dist_nopair], feed_dict=feed_dict)
        val_triplet_loss += val_err
        val_dpair += d_val_pair
        val_nopair += d_val_nopair

#
    loss_val_f = val_triplet_loss / val_iters
    d_pair_val_f = val_dpair / val_iters
    d_notpair_val_f = val_nopair / val_iters
    print("   val triplet loss: %f" % loss_val_f)
    print("   dist_pair: %f" % d_pair_val_f)
    print("   dist_no_pair: %f" % d_notpair_val_f)
    
    summary = tf.Summary()
    summary.value.add(tag="Loss",simple_value=loss_val_f)
    val_writer.add_summary(summary, iter_show)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        
        test_triplet_loss, test_dpair, test_nopair, batch_test_size = 0, 0, 0, 64
        test_iters = 741
        #test_iters = 2
        #test_street_loss, test_shop_loss, test_nopair_loss = 0, 0, 0
        #test_street_acc, test_shop_acc, test_nopair_acc = 0, 0, 0
        for iter_test in xrange(test_iters):
            street_test_batch, shop_test_batch, nopair_test_batch, y_street_test_batch, y_shop_test_batch, y_nopair_test_batch = input.load_test_images(batch_test_size)
            
            feed_dict = {x_street: street_test_batch, x_shop: shop_test_batch, x_nopair: nopair_test_batch, train_mode:False}
            
            test_err, d_test_pair, d_test_nopair = sess.run([triplet_loss, dist_pair, dist_nopair], feed_dict=feed_dict)
            test_triplet_loss += test_err
            test_dpair += d_test_pair
            test_nopair += d_test_nopair

#
        loss_out = test_triplet_loss / test_iters
        d_pair_out = test_dpair / test_iters
        d_notpair_out = test_nopair / test_iters
        print("   test triplet loss: %f" % loss_out)
        print("   dist_pair: %f" % d_pair_out)
        print("   dist_no_pair: %f" % d_notpair_out)
        with open('/ais/gobi4/fashion/retrieval/total_test_result.txt', 'a') as w_f:
            w_f.write(str(epoch+1)+'\t'+str(loss_out)+'\t'+str(d_pair_out)+'\t'+str(d_notpair_out)+'\n')
        w_f.close()

        street_network.save_npy(sess=sess,npy_path="/ais/gobi4/fashion/retrieval/street_alex.npy")
        shop_network.save_npy(sess=sess,npy_path="/ais/gobi4/fashion/retrieval/shop_alex.npy")

train_writer.close()
val_writer.close()
