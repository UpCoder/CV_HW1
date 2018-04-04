import numpy as np
from reader import Reader
from tools import extract_patches_from_images, extract_patch_from_image, split_array2array
import os
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
from reader import Reader
from multiprocessing import Pool
import scipy.io as scio


class Config:
    patch_size = 5
    patch_step = 2
    center_num = 128

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


def load_words(part_dir, rate=0.0003, save_path=None):

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


def get_KMeans_model(patches, save_path='./BoVW_model.m'):
    print("ok")
    time_start = time.time()
    patches = np.reshape(patches,[np.shape(patches)[0], -1])
    kmeans_obj = KMeans(Config.center_num)
    kmeans_obj.fit(patches)
    joblib.dump(kmeans_obj, save_path)
    time_end = time.time()
    print('Totally cost', time_end - time_start)


def get_features_sp(train_image_arr, train_label_arr, kmeans_obj, process_id=0):
    '''
    输入N个图像和其对应的label，获得学习得到的特征
    :param train_image_arr: [N, W, H, C]
    :param train_label_arr: [N]
    :return: [N, feature_size]
    '''
    train_features = []
    train_labels = []
    for idx, train_image in enumerate(train_image_arr):
        feature = np.zeros(Config.center_num)
        patches = extract_patch_from_image(train_image, Config.patch_size, Config.patch_step)
        if len(patches) == 0:
            continue
        patches = np.reshape(patches, [np.shape(patches)[0], -1])
        predicted_res = kmeans_obj.predict(patches)
        for i in range(len(predicted_res)):
            feature[predicted_res[i]] += 1
        train_features.append(feature)
        train_labels.append(train_label_arr[idx])
        if idx % 1000 == 0:
            print("Process %d / %d at P%d" % (idx, len(train_image_arr), process_id))
    return train_features, train_labels


def get_features(train_dir, test_path, model_path, processes_num=8, save_path='./BoVW_features.mat', reload=False):
    if reload and os.path.exists(save_path):
        save_dict = scio.loadmat(save_path)
        return save_dict['train_features'], save_dict['train_labels'], save_dict['test_features'], save_dict[
            'test_labels']

    def get_results(images_batches, labels_batches, kmeans_obj, processes_num):
        p = Pool()
        results = []
        for process_id in range(processes_num):
            result = p.apply_async(get_features_sp, args=(
                images_batches[process_id], labels_batches[process_id], kmeans_obj, process_id,))
            results.append(result)

        train_features = []
        train_labels = []
        for idx in range(processes_num):
            cur_train_features, cur_train_labels = results[idx].get()
            train_features.extend(cur_train_features)
            train_labels.extend(cur_train_labels)
        return train_features, train_labels
    reader = Reader(train_dir=train_dir, test_path=test_path)
    kmeans_obj = joblib.load(model_path)

    train_images_batches, train_labels_batches = split_array2array(processes_num, reader.train_images,
                                                                   reader.train_labels)
    test_images_batches, test_labels_batches = split_array2array(processes_num, reader.test_images, reader.test_labels)

    train_features, train_labels = get_results(train_images_batches, train_labels_batches, kmeans_obj, processes_num)
    print("train feature shape is ", np.shape(train_features))
    print("train label shape is ", np.shape(train_labels))
    test_features, test_labels = get_results(test_images_batches, test_labels_batches, kmeans_obj, processes_num)
    print("test feature shape is ", np.shape(test_features))
    print("test label shape is ", np.shape(test_labels))
    save_dict = {}
    save_dict['train_features'] = train_features
    save_dict['train_labels'] = train_labels
    save_dict['test_features'] = test_features
    save_dict['test_labels'] = test_labels
    scio.savemat(save_path, save_dict)
    return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    # 提取视觉单词
    # extract_words(
    #    '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
    #    '/home/give/homework/CV/dataset/affNIST/test.mat',
    #     '/home/give/homework/CV/dataset/affNIST/patches/'
    # )

    # 随机选取单词
    # patches = load_words('/home/give/homework/CV/dataset/affNIST/patches', save_path='./vocabulary.npy')
    # print(np.shape(patches))
    #
    # get_KMeans_model(patches)

    train_features, train_labels, test_features, test_labels = get_features(
        '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
        '/home/give/homework/CV/dataset/affNIST/test.mat',
        './BoVW_model.m',
        reload=True
    )
    from classification import SVM
    SVM.do(train_features, train_labels, test_features, test_labels)