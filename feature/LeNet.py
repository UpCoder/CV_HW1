import tensorflow as tf
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input
from reader import Reader
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from tools import calculate_acc_error


def LeNetModel(input_tensor):
    layer1_conv = Conv2D(filters=6, kernel_size=[5, 5], strides=(1, 1), activation='relu', padding='SAME')(input_tensor)
    layer2_pooling = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='SAME')(layer1_conv)
    layer3_conv = Conv2D(filters=6, kernel_size=[5, 5], strides=(1, 1), activation='relu', padding='SAME')(layer2_pooling)
    layer4_pooling = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='SAME')(layer3_conv)
    layer5_conv = Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], activation='relu', padding='SAME')(layer4_pooling)
    shape = layer5_conv.get_shape().as_list()
    flatten = Flatten()(layer5_conv)
    flatten.set_shape([None, shape[1]*shape[2]*shape[3]])
    layer6_fc1 = Dense(units=120, activation='relu')(flatten)
    layer7_fc2 = Dense(units=84, activation='relu')(layer6_fc1)
    layer8_fc3 = Dense(units=10, activation='relu')(layer7_fc2)
    return layer8_fc3


def train():
    reader = Reader(
        '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
        '/home/give/homework/CV/dataset/affNIST/test.mat'
    )
    input_tensor = Input((40, 40, 1))
    predicted = LeNetModel(input_tensor)
    model = Model(inputs=input_tensor, outputs=predicted)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='/home/give/homework/CV/hw1/feature/LeNet/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
    model.fit(np.expand_dims(reader.train_images, axis=3), keras.utils.to_categorical(reader.train_labels, 10), batch_size=128, epochs=10,
              validation_split=0.3, callbacks=[checkpoint])


def test():
    input_tensor = Input((40, 40, 1))
    predicted = LeNetModel(input_tensor)
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
    # train()
    # 测试模型
    test()
