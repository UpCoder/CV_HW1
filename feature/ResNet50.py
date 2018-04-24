# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v1
from reader import Reader
import keras
import numpy as np
import os
from tools import calculate_acc_error
tf.app.flags.DEFINE_string("save_model_path", "/home/give/homework/cv/CV_HW1/feature/save_model/ResNet/", "")
tf.app.flags.DEFINE_string("restore_model_path", "/home/give/homework/cv/CV_HW1/feature/save_model/ResNet/", "")
tf.app.flags.DEFINE_boolean("restore_flag", True, "")
FLAGS = tf.app.flags.FLAGS


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
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=predicted_tensor, labels=input_label_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)
    accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(input_label_tensor, axis=1),
                                           predictions=tf.argmax(tf.nn.softmax(predicted_tensor), axis=1))
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # hyper-parameters
    train_total = 1920000
    batch_size = 34
    epoch_num = 10

    # 输入数据
    reader = Reader(
        '/home/give/homework/cv/dataset/affiNist/training_and_validation_batches',
        '/home/give/homework/cv/dataset/affiNist/test.mat',
        batch_size=batch_size
    )
    print 'reader operation finished!'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if FLAGS.restore_flag:
            ckpt = tf.train.latest_checkpoint(FLAGS.restore_model_path)
            print('continue training from previous checkpoint from %s' % ckpt)
            start_step = int(os.path.basename(ckpt).split('-')[1])
            variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
            variable_restore_op(sess)
            sess.run(tf.assign(global_step, start_step))
        step = 0
        for epoch_id in range(epoch_num):
            start = 0
            while start < train_total:
                step += 1
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
                if step % 100 == 0:
                    print 'save model at ', FLAGS.save_model_path + 'model.ckpt'
                    saver.save(sess, FLAGS.save_model_path + 'model.ckpt', global_step=global_step)


def test():
    # tensor
    input_image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 40, 40, 1], name='image-input')
    predicted_tensor = ResNet50Model(input_image_tensor, is_training=True)
    predicted_tensor = tf.nn.softmax(predicted_tensor)
    global_step = tf.Variable(initial_value=0, trainable=False)


    # hyper-parameters
    test_total = 320000
    batch_size = 34
    epoch_num = 1

    # 输入数据
    reader = Reader(
        '/home/give/homework/cv/dataset/affiNist/training_and_validation_batches',
        '/home/give/homework/cv/dataset/affiNist/test.mat',
        batch_size=batch_size
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.latest_checkpoint(FLAGS.restore_model_path)
        print('continue training from previous checkpoint from %s' % ckpt)
        start_step = int(os.path.basename(ckpt).split('-')[1])
        variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
        variable_restore_op(sess)
        sess.run(tf.assign(global_step, start_step))
        start = 0
        predicted = []
        while start < test_total:
            end = start + batch_size
            if end > test_total:
                end = test_total
            cur_batch_images = reader.test_images[start:end]
            predicted_array = sess.run(tf.argmax(predicted_tensor, axis=1), feed_dict={
                input_image_tensor: np.expand_dims(cur_batch_images, axis=3)
            })
            predicted.extend(predicted_array)
            print 'Batch Accuracy[%d, %d] : %.4f' % (
            start, test_total, np.mean(np.asarray(predicted_array == reader.test_labels[start:end], np.float32)))
            start = end
        predicted = np.array(predicted)
        print 'Total Accuracy: ', np.mean(np.asarray(predicted == reader.test_labels, np.float32))
        calculate_acc_error(predicted, reader.test_labels)


if __name__ == '__main__':
    train()
    # test()