import tensorflow as tf
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input
from reader import Reader
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from tools import calculate_acc_error
from keras.backend import tensorflow_backend as ktf
from keras.layers import Lambda


def VGG16Model(input_tensor):
    input_tensor_resize = Lambda(lambda x: ktf.resize_images(x, 224, 224, 'channels_last'))(input_tensor)
    # Block1
    x = Conv2D(64, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block1_conv1')(input_tensor_resize)
    x = Conv2D(64, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pooling')(x)

    # Block2
    x = Conv2D(64, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pooling')(x)

    # Block3
    x = Conv2D(128, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(128, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block3_pooling')(x)

    # Block4
    x = Conv2D(256, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(256, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(256, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block4_pooling')(x)

    # Block5
    x = Conv2D(512, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), strides=[1, 1], activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block5_pooling')(x)

    shape = x.get_shape().as_list()
    flatten = Flatten()(x)
    flatten.set_shape([None, shape[1]*shape[2]*shape[3]])
    layer6_fc1 = Dense(units=4096, activation='relu')(flatten)
    layer7_fc2 = Dense(units=4096, activation='relu')(layer6_fc1)
    layer8_fc3 = Dense(units=10, activation='relu')(layer7_fc2)
    return layer8_fc3


def train():
    input_tensor = Input((40, 40, 1))
    predicted = VGG16Model(input_tensor)
    model = Model(inputs=input_tensor, outputs=predicted)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='/home/give/homework/CV/hw1/feature/LeNet/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
    reader = Reader(
        '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
        '/home/give/homework/CV/dataset/affNIST/test.mat'
    )
    model.fit(np.expand_dims(reader.train_images, axis=3), keras.utils.to_categorical(reader.train_labels, 10),
              batch_size=16, epochs=10,
              validation_split=0.3, callbacks=[checkpoint])


def test():
    input_tensor = Input((40, 40, 1))
    predicted = VGG16Model(input_tensor)
    model = Model(inputs=input_tensor, outputs=predicted)
    model.load_weights('/home/give/homework/CV/hw1/feature/LeNet/checkpoint-10-2.15.hdf5')
    reader = Reader(
        '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
        '/home/give/homework/CV/dataset/affNIST/test.mat'
    )
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    score = model.evaluate(np.expand_dims(reader.test_images, axis=3), keras.utils.to_categorical(reader.test_labels, 10), verbose=0)
    print('Total Test Accuracy is ', score[1])
    predict_res = model.predict(np.expand_dims(reader.test_images, axis=3))
    calculate_acc_error(np.argmax(predict_res, axis=1), reader.test_labels)

if __name__ == '__main__':
    # 训练模型
    train()
    # 测试模型
    # test()
