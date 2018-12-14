# Imports
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import gc

import LoadTFRecord as ltfr
from layers import *
from plotting import *
import dataset
import time
from tensorflow.contrib.tensorboard.plugins import projector

current_sec_time = lambda: int(round(time.time()))
sec_to_min = lambda sec: str(int(sec / 60)) + ':' + str(sec % 60)

# Train and test path
IN_DIR_TRAIN = 'C:/Users/rthun/BA/Data/100Classes128(0.75)/training/'
IN_DIR_TEST = 'C:/Users/rthun/BA/Data/100Classes128(0.75)/test/'

# Save Path
dir_path = "C:/Users/rthun/BA/CNN/cnn_models"
save_path = dir_path + "/model_1.3/model.ckpt"
tensorboard_path = dir_path + "/data/"
metadata = dir_path + '/model_1.3/metadata.tsv'
images_pca = 100

# Model Setup
size               = 128
depth              = 3
num_classes        = 10
max_size = 1000

# Weights
stddev               = 0.050
bias_weight          = 0.100

# Training Variables
epoch_errors        = []
batch_size          = 512
epochs              = 201

# Optimizer Setup
init_learning_rate = 1e-3
learning_rate       = init_learning_rate
target_learning_rate = 1e-4
learning_decay       = 0 #((init_learning_rate / target_learning_rate) - 1) / epochs
keep_prob = 0.5
l2_lambda = 1e-4

# Validation interval
validation_interval = 3

# Load Dataset
test_portion = 0.25
most_recent_classes, _ = ltfr.get_most_frequent_classes(num_classes, max_size=max_size)
data = dataset.read_train_sets(IN_DIR_TRAIN, size, most_recent_classes, max_size, test_portion, IN_DIR_TEST)
train_size = data.train.num_examples
test_size = data.valid.num_examples
dataset_size = train_size + test_size
print('Train and Test size:', train_size, test_size)

# Define the NN Model
def nn_train(num_conv, num_fc, hparam):
    graph = tf.Graph()
    with graph.as_default():
        # TF Placeholders
        with tf.name_scope('input'):
            is_training = tf.placeholder_with_default(False, shape=(), name='MODE')
            x = tf.placeholder(dtype=tf.float32, shape=[None, size, size, depth], name='x')
            y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='y')
            prob = tf.placeholder_with_default(1.0, shape=(), name='prob')

        # Define the Conv Neural Network
        def nn_model(x):
            net = tf.reshape(x, shape=[-1, size, size, 3])

            #conv layers
            net = conv_layer(net, depth, 32, stddev, bias_weight, k_size=3, name='conv1_1') #32
            net = conv_layer(net, 32, 32, stddev, bias_weight, k_size=3, name='conv1_2') #32
            net = max_pool_2x2(net, name='pool1')

            net = conv_layer(net, 32, 64, stddev, bias_weight, k_size=3, name='conv2_1') #64
            net = conv_layer(net, 64, 64, stddev, bias_weight, k_size=3, name='conv2_2') #64
            net = max_pool_2x2(net, name='pool2')

            net = conv_layer(net, 64, 128, stddev, bias_weight, k_size=3, name='conv3_1')  # 128
            net = conv_layer(net, 128, 128, stddev, bias_weight, k_size=3, name='conv3_2')  # 128
            net = conv_layer(net, 128, 128, stddev, bias_weight, k_size=3, name='conv3_3')  # 128
            net = max_pool_2x2(net, name='pool3')

            net = conv_layer(net, 128, 256, stddev, bias_weight, k_size=3, name='conv4_1') #256
            net = conv_layer(net, 256, 256, stddev, bias_weight, k_size=3, name='conv4_2') #256
            net = conv_layer(net, 256, 256, stddev, bias_weight, k_size=3, name='conv4_3') #256
            net = max_pool_2x2(net, name='pool4')

            net = create_flatten_layer(net)

            #fully connected layer
            net = fc_layer(net, net.get_shape()[1:4].num_elements(), 1024, stddev,
                          bias_weight, 'fc6')
            net = tf.nn.relu(net)
            net = dropout(net, prob, 'drop6')

            # net = fc_layer(net, 2024, 2024, stddev, #4096
            #                bias_weight, 'fc7')
            # net = tf.nn.relu(net)
            # net = dropout(net, prob, 'drop7')

            net = fc_layer(net, 1024, num_classes, stddev, #1024
                           bias_weight, 'fc8')

            return net


        with tf.name_scope('error'):
            pred_op = nn_model(x)
            y_pred = tf.nn.softmax(pred_op, name='y_pred')
            pred_op = tf.identity(pred_op)
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name]) * l2_lambda
            error_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=pred_op, labels=y) + lossL2, name="error_op")

        with tf.name_scope('train'):
            train_op = tf.cond(is_training, lambda: tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error_op), lambda: tf.Variable(False))

        with tf.name_scope('accuracy'):
            correct_result_op = tf.equal(
                tf.argmax(pred_op, 1), tf.argmax(y, 1))
            accuracy_op = tf.reduce_mean(
                tf.cast(correct_result_op, tf.float32))

        with tf.name_scope('confusion'):
            confusion_matrix_op = tf.confusion_matrix(
                tf.argmax(pred_op, 1),
                tf.argmax(y, 1))

        # with tf.name_scope('images'):
        #     image_reshape = tf.reshape(x, [-1, 128, 128 ,3])
        #     tf.summary.image('input', image_reshape, num_classes)

        # merge all summaries into a single op
        tf.summary.scalar("error_op", error_op)

        tf.summary.scalar("accuracy_op", accuracy_op)
        merged = tf.summary.merge_all()

    # Train the Neural Network
    with tf.Session(graph=graph) as sess:
        # images, labels, _, _ = data.valid.next_batch(images_pca)
        # images = tf.Variable(images, name='images')
        # images = tf.reshape(tensor=images, shape=[images_pca, 128 * 128 * 3])
        # sess.run(tf.global_variables_initializer())
        # images_run = sess.run(images)
        # images = tf.Variable(images_run)
        # sess.run(tf.global_variables_initializer())
        # with open(metadata, 'w') as metadata_file:
        #     for row in range(images_pca):
        #         c = np.nonzero(labels[::1])[1:][0][row]
        #         metadata_file.write('{}\n'.format(c))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter(tensorboard_path + 'VGGNet ' + hparam + ' num_classes=' + str(num_classes) + ' dataset_size=' + str(dataset_size) + ' decay=' + str(learning_decay) + ' dropout_prop=' + str(keep_prob) + ' l2_lambda=' + str(l2_lambda) + ' val_portion=' + str(test_portion) + ' epochs=' + str(epochs), graph=graph)
        # TRAINING
        print('Starting training!\n')
        print('With Train Size: ', train_size, ' and Testsize: ', test_size, '')
        print('With Params: ', hparam)
        print('Total number of parameters: ', get_total_number_of_parameters())
        global learning_rate
        for epoch in range(epochs):
            epoch_loss = 0.0
            t_start = current_sec_time()
            for i in range(int(train_size/batch_size)):
                x_batch, y_true_batch, _, _ = data.train.next_batch(batch_size)
                summary, _, cost, acc = sess.run(
                                            [merged, train_op, error_op, accuracy_op],
                                            feed_dict={x: x_batch, y: y_true_batch,
                                            is_training: True,
                                            prob: keep_prob})
                epoch_loss += cost
                writer.add_summary(summary, i)
            # learning_rate = 1 / (1 + learning_decay * (epoch + 1)) * init_learning_rate
            epoch_errors.append(epoch_loss)
            t_end = current_sec_time()
            print('Epoch ', epoch + 1, ' of ', epochs,
                    '\twith loss: ', epoch_loss,
                '\tand Accuracy: ', acc, '\tin: ', sec_to_min(t_end - t_start), 'min', learning_rate)
            if (epoch + 1) % validation_interval == 0:
                saver.save(sess, save_path)
                # learning_rate = learning_rate - decay
                print('Starting testing!\n')
                total_acc = 0
                for i in range(int(test_size / batch_size)):
                    x_batch, y_true_batch, _, _ = data.valid.next_batch(batch_size)
                    summary, _, cost, acc = sess.run(
                        [merged, train_op, error_op, accuracy_op],
                        feed_dict={x: x_batch, y: y_true_batch,
                                   is_training: False,
                                   prob: 1.0})
                    total_acc += acc
                total_acc /= int((test_size / batch_size))

                print('Acc: ', total_acc)
                write_accuracy(total_acc,
                               'VGGNet ' + hparam + ' num_classes=' + str(num_classes) + ' dataset_size=' + str(
                                   dataset_size) + ' conv_size=3' + ' dropout_prop=' + str(keep_prob) + ' l2_lambda=' + str(l2_lambda) + ' val_portion=' + str(test_portion) + ' epoch=' + str(epoch + 1), 'C:/Users/rthun/BA/CNN/cnn_models/data/accuracy.txt')

                # config = projector.ProjectorConfig()
                # embedding = config.embeddings.add()
                # embedding.tensor_name = images.name
                # embedding.metadata_path = metadata
                # projector.visualize_embeddings(writer, config)


def write_accuracy(acc, name, filename):
    with open(filename, "a") as text_file:
        text_file.write(name + ': ' + str(acc) + '\n')


def get_total_number_of_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def main():
    global learning_rate
    global epochs
    global keep_prob
    global l2_lambda
    num_conv = 10
    num_fc = 2

    hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    nn_train(num_conv, num_fc, hparam)

    # l2_lambda = 1e-3
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(num_conv, num_fc, hparam)

    # learning_rate = 1e-4
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(num_conv, num_fc, hparam)
    #
    # learning_rate = 5e-3
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(num_conv, num_fc, hparam)
    #
    # learning_rate = 5e-5
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(num_conv, num_fc, hparam)

    # keep_prob = 1.0
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)

    # epochs = 40
    # learning_rate = 1e-3
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)
    #
    # epochs = 150
    # learning_rate = 5e-3
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)
    #
    # epochs = 100
    # learning_rate = 1e-4
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)
    #
    # epochs = 100
    # learning_rate = 1e-5
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)
    #
    # epochs = 50
    # learning_rate = 5e-5
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)
    #
    # epochs = 50
    # learning_rate = 1e-6
    # hparam = make_hparam_string(learning_rate, num_conv, num_fc, batch_size)
    # nn_train(learning_rate, num_conv, num_fc, hparam)
    print('Done.')


if __name__ == "__main__":
    main()