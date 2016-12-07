import tensorflow as tf
#import tensorlayer as tl
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
import random
import bbox_input
import string

sess = tf.InteractiveSession()
x_street = tf.placeholder("float", [1, 227, 227, 3])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
#npy_path = '/ais/gobi4/fashion/retrieval/share_street_alex.npy'
#npy_path = '/ais/gobi4/fashion/retrieval/street_alex.npy'
street_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=False)

street_network.build(rgb=x_street, flag='street', train_mode=train_mode)


y_street = street_network.relu7

sess.run(tf.initialize_all_variables())
street_path = '/ais/gobi4/fashion/retrieval/alex_street_features.json'
img_path = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'
with open(street_path, 'w') as jsonfile:
    with open('/ais/gobi4/fashion/data/Cross-domain-Retrieval/list_test_pairs.txt', 'rb') as f:
        #data = random.sample(f.readlines(), 200)
        data = f.readlines()
        for line in data:
            line = line.split()
            x = input.load_image(img_path+line[0])
            x = x.reshape([1, 227, 227, 3])
            feed_dict = {x_street: x, train_mode: False}
            y = sess.run([y_street], feed_dict=feed_dict)
            y = np.asarray(y)
            jsondata = {'id': line[2], 'street_feature': y.tolist()}
            jsonfile.write(json.dumps(jsondata)+'\n')
    f.close()
jsonfile.close()    

