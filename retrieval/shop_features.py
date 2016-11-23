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
import json


sess = tf.InteractiveSession()
x_shop = tf.placeholder("float", [1, 227, 227, 3])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
shop_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=False)

shop_network.build(rgb=x_shop, flag="shop", train_mode=train_mode)

y_shop = shop_network.relu6

sess.run(tf.initialize_all_variables())
shop_path = '/ais/gobi4/fashion/retrieval/test_gallery.json'
img_path = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'
with open(shop_path, 'a') as jsonfile:
    with open('/ais/gobi4/fashion/data/Cross-domain-Retrieval/list_test_triplet_category.txt', 'rb') as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            x = input.load_image(img_path+line[3])
            x = x.reshape([1, 227, 227, 3])
            feed_dict = {x_shop: x, train_mode: False}
            y = sess.run([y_shop], feed_dict=feed_dict)
            y = np.asarray(y)
            jsondata = {'id': line[0], 'shop_feature': y.tolist()}
            jsonfile.write(json.dumps(jsondata)+'\n')
    f.close()
jsonfile.close()    

