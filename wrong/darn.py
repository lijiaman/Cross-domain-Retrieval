import tensorflow as tf
import numpy as np
import darn_shop
import darn_street
import utils
import retrieval_input

with tf.device('/gpu:0'):
    sess = tf.Session()

    street_imgs = tf.placeholder(tf.float32, [None, 227, 227, 3])
    shop_imgs = tf.placeholder(tf.float32, [None, 227, 227, 3])
    nopair_imgs = tf.placeholder(tf.float32, [None, 227, 227, 3])
    fc1_out = tf.placeholder(tf.float32, [None, 4096])
    train_mode = tf.placeholder(tf.bool)

    street_nin = darn_street.NIN()
    street_nin.build(street_imgs, train_mode)
    
    #shop_nin = darn_shop.NIN_SHOP()
    with tf.variable_scope("shop_imgs") as scope:
        shop_nin = darn_shop.NIN_SHOP()
        shop_nin.build(shop_imgs, train_mode)
        scope.reuse_variables()
        nopair_nin = darn_shop.NIN_SHOP()
        nopair_nin.build(nopair_imgs, train_mode)

    margin = 1
    loss = max(0, np.sqrt(np.sum((street_nin.fc1-shop_nin.fc1)**2))-np.sqrt(np.sum((street_nin.fc1-nopair_nin.fc1)**2))+margin)
    
    global_step = tf.Variable(0)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
    sess.run(tf.initialize_all_variables())
    for i in xrange(30000):
        street_batch,shop_batch,nopair_batch = retrieval_input.load_batchsize_images(batch_size=64)
        lr,cost,_ = sess.run([learning_rate,loss,optimizer], feed_dict={street_imgs: street_batch, shop_imgs: shop_batch, nopair_imgs: nopair_batch, train_mode: True})
        if i % 100 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print print_time
            print('Train iteration:{0}, loss:{1}, lr:{2}'.format(i, cost, lr))
               
    
 

    
