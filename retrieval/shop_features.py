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
import bbox_input
import string

sess = tf.InteractiveSession()
x_shop = tf.placeholder("float", [1, 227, 227, 3])
train_mode = tf.placeholder(tf.bool)

#npy_path = '/ais/gobi4/fashion/retrieval/share_shop_alex.npy'
#npy_path = '/ais/gobi4/fashion/retrieval/shop_alex.npy'
npy_path = '/ais/gobi4/fashion/data/alex_full.npy'
#npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
shop_network = alex.ALEXNET(alex_npy_path=npy_path, trainable=False)

shop_network.build(rgb=x_shop, flag='shop', train_mode=train_mode)

y_shop = shop_network.relu6

sess.run(tf.initialize_all_variables())
#shop_path = '/ais/gobi4/fashion/retrieval/test_gallery.json'
shop_path = '/ais/gobi4/fashion/retrieval/alex_full_test_gallery.json'
img_path = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'
with open(shop_path, 'w') as jsonfile:
    with open('/ais/gobi4/fashion/data/Cross-domain-Retrieval/list_test_pairs.txt', 'rb') as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            #print("line[3]:{0}".format(line[3]))            
            x = input.load_image(img_path+line[1])
            #x = input.load_image(img_path+line[2])
            x = x.reshape([1, 227, 227, 3])
            feed_dict = {x_shop: x, train_mode: False}
            y = sess.run([y_shop], feed_dict=feed_dict)
#            y, conv1, lrn1, pool1, conv2, lrn2, pool2, conv3, conv4, conv5,pool3, fc6, relu6_ori, fc7, relu7, fc8  = sess.run([y_shop, shop_network.conv1, shop_network.lrn1, shop_network.pool1, shop_network.conv2, shop_network.lrn2, shop_network.pool2, shop_network.conv3, shop_network.conv4, shop_network.conv5, shop_network.pool3, shop_network.fc6, shop_network.relu6, shop_network.fc7, shop_network.relu7, shop_network.fc8], feed_dict=feed_dict)
            y = np.asarray(y)
            jsondata = {'id': line[2], 'shop_feature': y.tolist()}
            jsonfile.write(json.dumps(jsondata)+'\n')

            #fc8 = np.asarray(fc8)
            #jsondata = {'fc8': fc8.tolist()}
            #jsonfile.write(json.dumps(jsondata)+'\n')

            #relu7 = np.asarray(relu7)
            #jsondata = {'relu7': relu7.tolist()}
            #jsonfile.write(json.dumps(jsondata)+'\n')

            #fc7 = np.asarray(fc7)
            #jsondata = {'fc7': fc7.tolist()}
            #jsonfile.write(json.dumps(jsondata)+'\n')

            #fc6 = np.asarray(fc6)
            #jsondata = {'fc6': fc6.tolist()}
            #jsonfile.write(json.dumps(jsondata)+'\n')

            #pool3 = np.asarray(pool3)
            #jsondata = {'pool3': pool3.tolist()}
            #jsonfile.write(json.dumps(jsondata)+'\n')
    f.close()
jsonfile.close()    

