import os
import tensorflow as tf

import numpy as np
import time
import inspect

NIN_MEAN = [103.939, 116.779, 123.68]


class NIN:
    """
    An implementation of Network in Network model.
    """

    def __init__(self, nin_npy_path=None, trainable=True):
        if nin_npy_path is not None:
            self.data_dict = np.load(nin_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the NIN
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [227, 227, 1]
        assert green.get_shape().as_list()[1:] == [227, 227, 1]
        assert blue.get_shape().as_list()[1:] == [227, 227, 1]
        bgr = tf.concat(3, [
            blue - NIN_MEAN[0],
            green - NIN_MEAN[1],
            red - NIN_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [227, 227, 3]

        self.conv1 = self.conv_layer(bgr, 3, 96, 11, 4, "conv1")
        self.cccp1 = self.cccp_layer(self.conv1, 96, 96, "cccp1")
        self.cccp2 = self.cccp_layer(self.cccp1, 96, 96, "cccp2")
        self.pool1 = self.max_pool(self.cccp2, 3, 2, "pool1")
        print "pool1.shape:"
        print self.pool1.get_shape()
        self.conv2 = self.conv_layer(self.pool1, 96, 256, 5, 1, "conv2",[2,2])
        self.cccp3 = self.cccp_layer(self.conv2, 256, 256, "cccp3")
        self.cccp4 = self.cccp_layer(self.cccp3, 256, 256, "cccp4")
        self.pool2 = self.max_pool(self.cccp4, 3, 2, "pool2")
        print "pool2.shape:"
        print self.pool2.get_shape()
        self.conv3 = self.conv_layer(self.pool2, 256, 384, 3, 1, "conv3",[1,1])
        self.cccp5 = self.cccp_layer(self.conv3, 384, 384, "cccp5")
        self.cccp6 = self.cccp_layer(self.cccp5, 384, 384, "cccp6")
        self.pool3 = self.max_pool(self.cccp6, 3, 2, "pool3")
        print "pool3.shape:"
        print self.pool3.get_shape()
        if train_mode is not None:
            self.pool3 = tf.cond(train_mode, lambda: tf.nn.dropout(self.pool3, 0.5), lambda: self.pool3)
        elif self.trainable:
            self.pool3 = tf.nn.dropout(self.pool3, 0.5)

        self.conv4_1024 = self.conv_layer(self.pool3, 384, 1024, 3, 1, "conv4_1024",[1,1])
        self.cccp7_1024 = self.cccp_layer(self.conv4_1024, 1024, 1024, "cccp7_1024")
        self.cccp8_1024 = self.cccp_layer(self.cccp7_1024, 1024, 50, "cccp8_1024")#1000 is number of categories
        self.pool4 = self.avg_pool(self.cccp8_1024, 6, 1, "pool4")
        print "pool4.shape:"
        print self.pool4.get_shape()
        self.final = tf.reshape(self.pool4, [-1,50])
        print "final.shape:"
        print self.final.get_shape()
        self.prob = tf.nn.softmax(self.final, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, filter_size, stride, name):
        return tf.nn.avg_pool(bottom, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def max_pool(self, bottom, filter_size, stride, name):
        return tf.nn.max_pool(bottom, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, filter_size, stride, name, pad=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
	    if pad is not None:
	        bottom = tf.pad(bottom, [[0,0],[pad[0],pad[0]],[pad[1],pad[1]],[0,0]],"CONSTANT")
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            print name
            print relu.get_shape()
            return relu

    def cccp_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(1, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            print name
            print relu.get_shape()
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./nin-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
