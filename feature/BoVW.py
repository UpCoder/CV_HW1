import numpy as np
from reader import Reader
from tools import extract_patches_from_images
import os


class Config:
    patch_size = 5
    patch_step = 2


def extract_words(train_data_dir, test_data_path, save_dir):
    '''
    提取所有的视觉单词，并将视觉单词保存成numpy的格式
    :param train_data_dir:训练集的所在的文件夹
    :param test_data_path: 测试数据的路径
    :param save_path: 保存的路径
    :return:
    '''
    reader = Reader(train_dir=train_data_dir, test_path=test_data_path)
    extract_patches_from_images(reader.train_images, Config.patch_size, Config.patch_step, save_dir)


def load_words(part_dir, rate=0.003, save_path=None):

    names = os.listdir(part_dir)
    patches = []
    for name in names:
        print(name)
        path = os.path.join(part_dir, name)
        cur_patch = np.load(path)
        indexs = list(range(len(cur_patch)))
        np.random.shuffle(indexs)
        indexs = indexs[:int(rate*len(indexs))]
        patches.extend(cur_patch[indexs])
    if save_path is not None:
        np.save(save_path, patches)
    return patches

def get_KMeans_model(patches, center_num=128):

if __name__ == '__main__':
    # 提取视觉单词
    # extract_words(
    #    '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
    #    '/home/give/homework/CV/dataset/affNIST/test.mat',
    #     '/home/give/homework/CV/dataset/affNIST/patches/'
    # )
    # 随机选取单词
    patches = load_words('/home/give/homework/CV/dataset/affNIST/patches', save_path='./vocabulary.npy')
    print(np.shape(patches))