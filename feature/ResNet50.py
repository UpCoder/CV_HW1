# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v1
from reader import Reader
import keras
import numpy as np


def ResNet50Model(input_tensor, weight_decay=1e-5, is_training=True):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        input_tensor = tf.image.resize_images(input_tensor, [224, 224])
        logits, end_points = resnet_v1.resnet_v1_50(input_tensor, is_training=is_training, scope='resnet_v1_50')
        feature = tf.reduce_mean(logits, reduction_indices=[1, 2])
        fc1 = tf.contrib.layers.fully_connected(feature, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=10)
    return fc2


def train():
    print 'train'
    # 输入的tensor
    input_image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 40, 40, 1], name='image-input')
    input_label_tensor = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='label-input')
    global_step = tf.Variable(initial_value=0, trainable=False)
    predicted_tensor = ResNet50Model(input_image_tensor)
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_tensor, labels=input_label_tensor)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(input_label_tensor),
                                   predictions=tf.argmax(tf.nn.softmax(predicted_tensor)))

    # hyper-parameters
    train_total = 1920000
    batch_size = 32
    epoch_num = 10

    # 输入数据
    reader = Reader(
        '/home/ld/dataset/affNIST/training_and_validation_batches',
        '/home/ld/dataset/affNIST/test.mat',
        batch_size=batch_size
    )
    print 'reader operation finished!'
    with tf.Session() as sess:
        for epoch_id in range(epoch_num):
            start = 0
            while start < train_total:
                end = start + batch_size
                if end > train_total:
                    end = train_total
                cur_batch_images = reader.train_images[start:end]
                cur_batch_labels = reader.train_labels[start:end]
                feed_dict = {
                    input_image_tensor: np.expand_dims(cur_batch_images, axis=3),
                    input_label_tensor: keras.utils.to_categorical(cur_batch_labels, num_classes=10)
                }
                _, train_loss, train_acc = sess.run([train_op, cross_entropy_loss, accuracy], feed_dict=feed_dict)
                print '[%d --> %d] / %d, training loss: %.4f, training acc: %.4f' % (
                start, end, train_total, train_loss, train_acc)
                start = end

if __name__ == '__main__':
    # input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 1], name='x-input')
    #
    # ResNet50Model(input_tensor)
    train()