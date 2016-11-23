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
ALEX_MEAN = [103.939, 116.779, 123.68]

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

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

net = np.load("/ais/gobi4/fashion/bvlc_alexnet.npy").item()
conv1_w = tf.Variable(net["conv1"][0])
print conv1_w

def AlexNet(rgb):
    
    net = np.load("/ais/gobi4/fashion/bvlc_alexnet.npy").item()
    conv1_w_init = tf.constant_initializer(value=net["conv1"][0],dtype=tf.float32)
    conv1_w_shape = net["conv1"][0].shape
    conv1_w = tf.Variable(net["conv1"][0])
    conv1_b = tf.Variable(net["conv1"][1])
    print conv1_w
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    red, green, blue = tf.split(3, 3, rgb_scaled)
    assert red.get_shape().as_list()[1:] == [227, 227, 1]
    assert green.get_shape().as_list()[1:] == [227, 227, 1]
    assert blue.get_shape().as_list()[1:] == [227, 227, 1]
    bgr = tf.concat(3, [blue - ALEX_MEAN[0], green - ALEX_MEAN[1], red - ALEX_MEAN[2]])
    assert bgr.get_shape().as_list()[1:] == [227, 227, 3]
    """ input layer """
    net_in = tl.layers.InputLayer(bgr, name='input_layer')
    """ conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                    act = tf.nn.relu,
                    shape = [11, 11, 3, 96],
                    strides = [1, 4, 4, 1],
                    padding='SAME',
                    W_init = conv1_w,
                    b_init = conv1_b,
                    name ='conv1')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool1')
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [5, 5, 96, 256],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    W_init = tf.Variable(net['conv2'][0]),
                    b_init = tf.Variable(net['conv2'][1]),
                    name ='conv2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool2')
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 384],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    W_init = tf.convert_to_tensor(net['conv3'][0], dtype=tf.float32),
                    b_init = tf.convert_to_tensor(net['conv3'][1], dtype=tf.float32),
                    name ='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 384, 384],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    W_init = tf.convert_to_tensor(net['conv4'][0], dtype=tf.float32),
                    b_init = tf.convert_to_tensor(net['conv4'][1], dtype=tf.float32),
                    name ='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 384, 256],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    W_init = tf.convert_to_tensor(net['conv5'][0], dtype=tf.float32),
                    b_init = tf.convert_to_tensor(net['conv5'][1], dtype=tf.float32),
                    name ='conv3_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool3')
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    """ fc 6~8 """
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DenseLayer(network, n_units=4096, act = tf.tanh, W_init=tf.convert_to_tensor(net['fc6'][0], dtype=tf.float32), b_init=tf.convert_to_tensor(net['fc6'][1], dtype=tf.float32),name = 'fc6')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=4096, act = tf.tanh, W_init=tf.convert_to_tensor(net['fc7'][0], dtype=tf.float32), b_init=tf.convert_to_tensor(net['fc7'][1], dtype=tf.float32), name = 'fc7')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=50, act = tf.identity, W_init=tf.convert_to_tensor(net['fc7'][0], dtype=tf.float32), b_init=tf.convert_to_tensor(net['fc7'][1], dtype=tf.float32), name = 'fc8')

    print("build model finished: %fs" % (time.time() - start_time))
    return network

model_file_name = "finetune_alex_deepfashion.ckpt"
resume = False
sess = tf.InteractiveSession()
batch_size = 128
x = tf.placeholder("float", [batch_size, 227, 227, 3])
y_ = tf.placeholder(tf.int32, shape=[batch_size,])
network = AlexNet(x)
y = network.outputs
ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
cost = ce
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, y_, 5, 'top-5'), tf.float32))
top10_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, y_, 10, 'top-10'), tf.float32))
tf.scalar_summary('Loss', cost)
tf.scalar_summary('Top1-Acc', acc)
tf.scalar_summary('Top5-Acc', top5_acc)
tf.scalar_summary('Top10-Acc', top10_acc)

merged = tf.merge_all_summaries()
log_dir = 'finetune_visual'
train_writer = tf.train.SummaryWriter(log_dir+'/train')
val_writer = tf.train.SummaryWriter(log_dir+'/val')
# train
n_epoch = 50
global_step = tf.Variable(0)
starter_learning_rate = 0.000001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1635, 0.96, staircase=True)
print_freq = 1

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, global_step=global_step, var_list=train_params)
#train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)

sess.run(tf.initialize_all_variables())
if resume:
    print("Load existing model " + "!"*10)
    saver = tf.train.Saver()
    path = "/ais/gobi4/fashion/data/"
    saver.restore(sess, path+model_file_name)

network.print_params()
network.print_layers()

#print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
ISOTIMEFORMAT = '%Y-%m-%d %X'
iter_show = 0
for epoch in range(n_epoch):
    start_time = time.time()
    iter_per_epoch = 1635
    for iter in xrange(iter_per_epoch):
    	x_batch, y_batch = input.load_batchsize_images(batch_size)
        feed_dict = {x: x_batch, y_: y_batch}
        feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
        _, err, ac, top5_ac, top10_ac, lr, train_summary = sess.run([train_op, cost, acc, top5_acc, top10_acc, learning_rate, merged], feed_dict=feed_dict)
        iter_show += 1
        train_writer.add_summary(train_summary, iter_show)
        if iter % 10 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Train Loss: %f" % err)
            print("   Top1 Acc: %f" % ac)
            print("   Top5 Acc: %f" % top5_ac)
            print("   Top10 Acc: %f" % top10_ac)

        x_val_batch, y_val_batch = input.load_val_images(batch_size)
        dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
        feed_dict = {x: x_val_batch, y_: y_val_batch}
        feed_dict.update(dp_dict)
        val_err, val_ac, val_top5_ac, val_top10_ac, val_summary = sess.run([cost, acc, top5_acc, top10_acc, merged], feed_dict=feed_dict)
        val_writer.add_summary(val_summary, iter_show)  
        if iter % 10 == 0:      
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Val iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Val Loss: %f" % val_err)
            print("   Val Top1 Acc: %f" % val_ac)
            print("   Val Top5 Acc: %f" % val_top5_ac)
            print("   Val Top10 Acc: %f" % val_top10_ac)


    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, batch_size = 0, 0, 128
        #for iter in xrange(iter_per_epoch):
    	    #x_batch, y_batch = input.load_batchsize_images(batch_size)
            #dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
            #feed_dict = {x: x_batch, y_: y_batch}
            #feed_dict.update(dp_dict)
            #err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            #assert not np.isnan(err), 'Model diverged with cost = NaN'
            #train_loss += err; train_acc += ac; 
       
        #print("   train loss: %f" % (train_loss/ iter_per_epoch))
        #print("   train acc: %f" % (train_acc/ iter_per_epoch))
        test_loss, test_acc, test_top5_acc, test_top10_acc, batch_test_size = 0, 0, 0, 0, 128
        test_iters = 312
        for iter_test in xrange(test_iters):
            #print iter_test
            x_test_batch, y_test_batch = input.load_test_images(batch_test_size)
            dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
            feed_dict = {x: x_test_batch, y_: y_test_batch}
            feed_dict.update(dp_dict)
            err, ac, top5_ac, top10_ac = sess.run([cost, acc, top5_acc, top10_acc], feed_dict=feed_dict)
            test_loss += err; test_acc += ac; test_top5_acc += top5_ac; test_top10_acc += top10_ac
        print("   Test Loss: %f" % (test_loss/ test_iters))
        print("   Test Top1-Acc: %f" % (test_acc/ test_iters))
        print("   Test Top5-Acc: %f" % (test_top5_acc / test_iters))
        print("   Test Top10-Acc: %f" % (test_top10_acc / test_iters))

    if (epoch + 1) % 1 == 0:
        print("Save model " + "!"*10);
        saver = tf.train.Saver()
        file_path = "/ais/gobi4/fashion/data/"
        save_path = saver.save(sess, file_path+model_file_name)

train_writer.close()
val_writer.close()
