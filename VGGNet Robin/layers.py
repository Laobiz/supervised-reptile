# Imports
import tensorflow as tf
import numpy as np


#Inception
def inception_layer_1(x, size_in, k_size_1x1, k_size3x3, k_size5x5, stddev, bias_weight, name):
    conv1x1_1 = conv_layer(x, size_in, k_size_1x1, stddev, bias_weight, 1, relu=False, name=name + '_conv1x1_1')

    conv1x1_2 = conv_layer(x, size_in, k_size_1x1, stddev, bias_weight, 1, name=name + '_conv1x1_2')
    conv3x3_2 = conv_layer(conv1x1_2, k_size_1x1, k_size3x3, stddev, bias_weight, 3, relu=False, name=name + '_conv3x3_2')

    conv1x1_3 = conv_layer(x, size_in, k_size_1x1, stddev, bias_weight, 1, name=name + '_conv1x1_3')
    conv3x3_3 = conv_layer(conv1x1_3, k_size_1x1, k_size5x5, stddev, bias_weight, 3, name=name + '_conv3x3_3')
    conv3x3_4 = conv_layer(conv3x3_3, k_size5x5, k_size5x5, stddev, bias_weight, 3, relu=False, name=name + '_conv3x3_4')

    pool2x2_4 = average_pool_2x2(x, name + '_pool2x2_4')
    conv1x1_4 = conv_layer(pool2x2_4, size_in, k_size_1x1, stddev, bias_weight, 1, relu=False, name=name + '_conv1x1_4')

    return tf.nn.relu(tf.concat([conv1x1_1, conv3x3_2, conv3x3_4, conv1x1_4], axis=3))


#ResLayer
def res_layer():
    pass



# Max Pooling
def max_pool_2x2(x, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(x,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',)

def max_pool_3x3(x, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(x,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 3, 3, 1],
                              padding='SAME', )

def max_pool(x, k_size, strides, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(x,
                              ksize=[1, k_size, k_size, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME', )


# Average Pooling
def average_pool_2x2(x, name):
    with tf.name_scope(name):
        return tf.nn.avg_pool(x,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1],
                    padding='SAME')


def average_pool(x, k_size, strides, name):
    with tf.name_scope(name):
        return tf.nn.avg_pool(x,
                    ksize=[1, k_size, k_size, 1],
                    strides=[1, strides, strides, 1],
                    padding='SAME')


# Convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, 
                strides=[1, 1, 1, 1], 
                padding='SAME',)


# Dropout
def dropout(x, keep_prob, name):
    with tf.name_scope(name):
        return tf.nn.dropout(x, keep_prob=keep_prob)


# Define a convolutional layer
def conv_layer(x, 
                size_in, 
                size_out, 
                stddev, 
                bias_weight,
                k_size=3,
                relu = True,
                name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([k_size, k_size, size_in, size_out], stddev=stddev), name='weight')
        b = tf.Variable(tf.constant(bias_weight, shape=[size_out]), name='bias')
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        conv = conv2d(x, W)
        add = tf.add(conv, b)
        act = add
        if relu:
            act = tf.nn.relu(add)
        return act


# Define a Fully-connected Layer
def fc_layer(x, 
            size_in, 
            size_out, 
            stddev,
            bias_weight, 
            name="fc"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=stddev), name='weight')
        b = tf.Variable(tf.constant(bias_weight, shape=[size_out]), name='bias')
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        val = tf.add(tf.matmul(x, W), b)
        return val


# Define Flatten/Dense function
def dense(x, reshape):
    return tf.reshape(x, [-1, reshape])


def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer