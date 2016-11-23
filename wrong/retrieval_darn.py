import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np
import time
import inspect
import skimage
import skimage.io
import skimage.transform
import retrieval_input


def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1

def NIN(bgr,flag,reuse=None):
    with tf.variable_scope(flag, reuse=reuse):
        start_time = time.time()
        print("build model started")
    
        tl.layers.set_name_reuse(reuse)
        """ input layer """
        net_in = tl.layers.InputLayer(bgr, name= 'input_layer')
        """ conv1 """
        network = tl.layers.Conv2dLayer(net_in,
                        act = tf.nn.relu,
                        shape = [7, 7, 3, 96],
                        strides = [1, 2, 2, 1],
                        padding='SAME',
                        name = 'conv1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 96, 96],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 96, 96],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp2')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name = 'pool1')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 96, 256],
                        strides = [1, 2, 2, 1],
                        padding='SAME',
                        name = 'conv2')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 256, 256],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp3')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 256, 256],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp4')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name = 'pool2')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 256, 512],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'conv3')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 512, 512],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp5')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 512, 512],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp6')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 512, 1024],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'conv4')
        extra_conv4 = tl.layers.PoolLayer(network,
                        ksize=[1, 5, 5, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name = 'extra1')
        extra_conv4 = tl.layers.FlattenLayer(extra_conv4, name= 'flatten_ex4')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 1024, 1024],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp7')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 1024, 512],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp8')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 512, 384],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp9')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [3, 3, 384, 512],
                        strides = [1, 2, 2, 1],
                        padding='SAME',
                        name = 'conv5')
        extra_conv5 = tl.layers.PoolLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name = 'extra2')
        extra_conv5 = tl.layers.FlattenLayer(extra_conv5, name= 'flatten_ex5')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 512, 512],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp10')
        network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [1, 1, 512, 512],
                        strides = [1, 1, 1, 1],
                        padding='SAME',
                        name = 'cccp11')
        network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name = 'pool3')
        
        
        
        """ fc 6~8 """
        network = tl.layers.FlattenLayer(network, name= 'flatten')
        fc1 = tl.layers.DenseLayer(network, n_units=4096, act = tf.tanh, name =  'fc6')
        fc2 = tl.layers.DenseLayer(fc1, n_units=4096, act = tf.tanh, name =  'fc7')
        fc3 = tl.layers.DenseLayer(fc2, n_units=23, act = tf.tanh, name =  'fc8')
        fc1 = tl.layers.ConcatLayer(layer = [fc1, extra_conv4, extra_conv5], name = 'concat_layer')
        network = fc3
        print("build model finished: %fs" % (time.time() - start_time))
        return fc1.outputs, fc3.outputs, network


model_file_name = "toy200_retrieval_darn.ckpt"
resume = False
sess = tf.InteractiveSession()
batch_size = 32
x_street = tf.placeholder("float", [batch_size, 227, 227, 3])
y_street_ = tf.placeholder(tf.int32, shape=[batch_size,])
x_shop = tf.placeholder("float", [batch_size, 227, 227, 3])
y_shop_ = tf.placeholder(tf.int32, shape=[batch_size,])
x_nopair = tf.placeholder("float", [batch_size, 227, 227, 3])
y_nopair_ = tf.placeholder(tf.int32, shape=[batch_size,])


fc1_street, fc3_street, street_nin = NIN(x_street,'street')
fc1_shop, fc3_shop, shop_nin = NIN(x_shop,'shop')
fc1_nopair, fc3_nopair, nopair_nin = NIN(x_nopair,'shop',reuse=True)

dist_pair = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(fc1_street,fc1_shop))))
dist_nopair = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(fc1_street,fc1_nopair))))
#dist_pair = np.sqrt(np.sum((fc1_street - fc1_shop)**2))
#dist_nopair = np.sqrt(np.sum((fc1_street - fc1_nopair)**2))
zero = tf.constant(0,dtype="float")
margin = tf.constant(0.3)
dist = tf.add(tf.sub(dist_pair,dist_nopair),margin)
pred = tf.less(dist,zero)
triplet_loss = tf.select(pred,zero,dist)
#triple_loss = max(0, 0.3 + dist_pair - dist_nopair)

#For street image category prediction
y_street = fc3_street
y_shop = fc3_shop
y_nopair = fc3_nopair
street_category_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_street, y_street_))
street_correct_prediction = tf.equal(tf.cast(tf.argmax(y_street, 1), tf.int32), y_street_)
street_category_acc = tf.reduce_mean(tf.cast(street_correct_prediction, tf.float32))

shop_category_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_shop, y_shop_))
shop_correct_prediction = tf.equal(tf.cast(tf.argmax(y_shop, 1), tf.int32), y_shop_)
shop_category_acc = tf.reduce_mean(tf.cast(shop_correct_prediction, tf.float32))

nopair_category_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_nopair, y_nopair_))
nopair_correct_prediction = tf.equal(tf.cast(tf.argmax(y_nopair, 1), tf.int32), y_nopair_)
nopair_category_acc = tf.reduce_mean(tf.cast(nopair_correct_prediction, tf.float32))



# train
n_epoch = 5000
global_step = tf.Variable(0)
start_triplet_lr = 0.00001
start_street_category_lr = 0.00001
start_shop_category_lr = 0.00001
start_nopair_category_lr = 0.00001
triplet_lr = tf.train.exponential_decay(start_triplet_lr, global_step, 10, 0.96, staircase=True)
street_category_lr = tf.train.exponential_decay(start_street_category_lr, global_step, 10, 0.96, staircase=True)
shop_category_lr = tf.train.exponential_decay(start_shop_category_lr, global_step, 10, 0.96, staircase=True)
nopair_category_lr = tf.train.exponential_decay(start_nopair_category_lr, global_step, 10, 0.96, staircase=True)
print_freq = 1

street_params = street_nin.all_params
shop_params = shop_nin.all_params
train_params = street_params+shop_params
train_op = tf.train.AdamOptimizer(triplet_lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(triplet_loss, var_list=train_params)

street_category_op = tf.train.AdamOptimizer(street_category_lr, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(street_category_cost, var_list=train_params)
shop_category_op = tf.train.AdamOptimizer(shop_category_lr, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(shop_category_cost, var_list=train_params)
nopair_category_op = tf.train.AdamOptimizer(nopair_category_lr, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(nopair_category_cost, var_list=train_params)


sess.run(tf.initialize_all_variables())
if resume:
    print("Load existing model " + "!"*10)
    saver = tf.train.Saver()
    file_path = "/ais/gobi4/fashion/data/"
    saver.restore(sess, file_path+model_file_name)

street_nin.print_params()
street_nin.print_layers()

shop_nin.print_params()
shop_nin.print_layers()

nopair_nin.print_params()
nopair_nin.print_layers()
print('   batch_size: %d' % batch_size)

for epoch in range(n_epoch):
    start_time = time.time()
    iter_per_epoch = 8
    for iter in xrange(iter_per_epoch):
    	street_batch, shop_batch, nopair_batch, y_street_batch, y_shop_batch, y_nopair_batch = retrieval_input.load_batchsize_images(batch_size)
        #print "street_batch.shape:"
        #print street_batch.shape
        #print "iter:"
        #print iter
        feed_dict = {x_street: street_batch, x_shop: shop_batch, x_nopair: nopair_batch, y_street_: y_street_batch, y_shop_: y_shop_batch, y_nopair_: y_nopair_batch}
        feed_dict.update( dict(street_nin.all_drop.items()+shop_nin.all_drop.items()) )        # enable all dropout/dropconnect/denoising layers
        err, d_pair, d_nopair, _, street_loss, street_acc, _, shop_loss, shop_acc, _, nopair_loss, nopair_acc, t_lr, street_lr, shop_lr, nopair_lr = sess.run([triplet_loss, dist_pair, dist_nopair, street_category_op, street_category_cost, street_category_acc, shop_category_op, shop_category_cost, shop_category_acc, nopair_category_op, nopair_category_cost, nopair_category_acc, triplet_lr, street_category_lr, shop_category_lr, nopair_category_lr ], feed_dict=feed_dict)
        if iter % 4 == 0:
            print("{0} Epoch: {1}, Train iteration: {2}".format(time.time(), epoch+1, iter+1))
            #print("   triplet loss: %f" % err)
            #print("   dist_pair: %f" % d_pair)
            #print("   dist_no_pair: %f" % d_nopair)
            print("Street category predict loss:{0}, acc:{1}, lr:{2}".format(street_loss, street_acc, street_lr))
            print("Shop category predict loss:{0}, acc:{1}, lr:{2}".format(shop_loss, shop_acc, shop_lr))
            print("Nopair category predict loss:{0}, acc:{1}, lr:{2}".format(nopair_loss, nopair_acc, nopair_lr))

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        
        test_triplet_loss, test_dpair, test_nopair, batch_test_size = 0, 0, 0, 32
        test_iters = 8
        test_street_loss, test_shop_loss, test_nopair_loss = 0, 0, 0
        test_street_acc, test_shop_acc, test_nopair_acc = 0, 0, 0
        for iter_test in xrange(test_iters):
            street_test_batch, shop_test_batch, nopair_test_batch, y_street_test_batch, y_shop_test_batch, y_nopair_test_batch = retrieval_input.load_test_images(batch_test_size)
            dp_dict = tl.utils.dict_to_one( dict(street_nin.all_drop.items()+shop_nin.all_drop.items()) )    # disable all dropout/dropconnect/denoising layers
            feed_dict = {x_street: street_test_batch, x_shop: shop_test_batch, x_nopair: nopair_test_batch, y_street_: y_street_test_batch, y_shop_: y_shop_test_batch, y_nopair_: y_nopair_test_batch}
            feed_dict.update(dp_dict)
            err, d_pair, d_nopair, street_cost, street_ac, shop_cost, shop_ac, nopair_cost, nopair_ac, t_test_lr, street_test_lr, shop_test_lr, nopair_test_lr = sess.run([triplet_loss, dist_pair, dist_nopair, street_category_cost, street_category_acc, shop_category_cost, shop_category_acc, nopair_category_cost, nopair_category_acc, triplet_lr, street_category_lr, shop_category_lr, nopair_category_lr], feed_dict=feed_dict)
            test_triplet_loss += err
            test_dpair += d_pair
            test_nopair += d_nopair
            test_street_loss += street_cost
            test_shop_loss += shop_cost
            test_nopair_loss += nopair_cost
            test_street_acc += street_ac
            test_shop_acc += shop_ac
            test_nopair_acc += nopair_ac

        #print("   test triplet loss: %f" % (test_triplet_loss/ test_iters))
        #print("   dist_pair: %f" % (test_dpair/ test_iters))
        #print("   dist_no_pair: %f" % (test_nopair/ test_iters))
        print("   test street loss:{0}, acc:{1}, lr:{2}".format(test_street_loss, test_street_acc/test_iters, street_test_lr))
        print("   test shop loss:{0}, acc:{1}, lr:{2}".format(test_shop_loss, test_shop_acc/test_iters, shop_test_lr))
        print("   test nopair loss:{0}, acc:{1}, lr:{2}".format(test_nopair_loss, test_nopair_acc/test_iters, nopair_test_lr))
    if (epoch + 1) % 1 == 0:
        print("Save model " + "!"*10);
        saver = tf.train.Saver()
        file_path = "/ais/gobi4/fashion/data/"
        save_path = saver.save(sess, file_path+model_file_name)

