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
import json
import random
import bbox_input
import string

sess = tf.InteractiveSession()
x_street = tf.placeholder("float", [1, 227, 227, 3])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/data/alex_full.npy'
#npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
#npy_path = '/ais/gobi4/fashion/retrieval/street_alex.npy'
street_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=False)

street_network.build(rgb=x_street, flag="street", train_mode=train_mode)


y_street = street_network.relu6

sess.run(tf.initialize_all_variables())
street_path = '/ais/gobi4/fashion/retrieval/alex_full_street_features.json'
img_path = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'
with open(street_path, 'w') as jsonfile:
    with open('/ais/gobi4/fashion/data/Cross-domain-Retrieval/test_pairs_category.txt', 'rb') as f:
        #data = random.sample(f.readlines(), 200)
        data = f.readlines()
        for line in data:
            line = line.split()
           # x1 = string.atoi(line[3])
           # y1 = string.atoi(line[4])
           # x2 = string.atoi(line[5])
           # y2 = string.atoi(line[6])
           # x = bbox_input.load_image(img_path+line[1], x1, y1, x2, y2)
           # x = x.reshape([1, 227, 227, 3])
            street_path = line[1]
            x = input.load_image(street_path)
            feed_dict = {x_street: x, train_mode: False}
            y = sess.run([y_street], feed_dict=feed_dict)
            y = np.asarray(y)
            jsondata = {'id': line[0], 'street_feature': y.tolist()}
            jsonfile.write(json.dumps(jsondata)+'\n')
    f.close()
jsonfile.close()    

