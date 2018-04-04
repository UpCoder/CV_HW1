# -*- coding=utf-8 -*-
from PIL import Image
import numpy as np
import os
import gc


def show_image(image_arr, resize=None):
    img = Image.fromarray(image_arr)
    if resize is not None:
        img = img.resize(resize)
    img.show()


def extract_patch_from_image(image, patch_size, patch_step, zero=True):
    '''
    从一张图片中提取patch，返回N,pach_size, patch_size格式的数组
    :param image: 图像，numpy array对象
    :param patch_size: patch的大小，正奇数
    :param patch_step: 提取patch的步长，正整数
    :param zero: 是否允许patch里面是全0
    :return: [N, patch_size, patch_size]格式的数组

    '''
    patches = []
    w, h = np.shape(image)
    patch_size_half = int(patch_size / 2)
    if zero:
        xs, ys = np.where(image != 0)
        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)
    else:
        min_x = 0
        max_x = w
        min_y = 0
        max_y = h
    for i in range(min_x, max_x, patch_step):
        for j in range(min_y, max_y, patch_step):
            cur_patch = image[i-patch_size_half:i+patch_size_half+1, j-patch_size_half:j+patch_size_half+1]
            cur_shape = np.shape(cur_patch)
            if cur_shape[0] != patch_size or cur_shape[1] != patch_size:
                # print('size not equal')
                continue
            patches.append(cur_patch)
    return patches


def extract_patches_from_images(images, patch_size, patch_step, save_dir=None):
    '''
    从多幅图像中提取patch，返回N,patch_size,patch_size格式的数组
    :param images:
    :param patch_size:
    :param patch_step:
    :return:
    '''
    patches = []
    counter = 1
    for indx, image in enumerate(images):
        if indx % 10000 == 0:
            print('Processed %d / %d' % (indx, len(images)))
        if len(patches) > 1e6 and save_dir is not None:
            save_path = os.path.join(save_dir, 'part_'+str(counter)+'.npy')
            print('Saveing %d patches at %s' % (len(patches), save_path))
            np.save(save_path, patches)
            patches = []
            counter += 1
            gc.collect()
        patches.extend(extract_patch_from_image(image, patch_size, patch_step))
    return patches


def split_array2array(*par):
    '''
    将一个或者是多个数组拆分成n个
    :param par: N, 原来的一个或者是多个数组
    :return:
    '''
    par = list(par)
    for i in range(1, len(par)):
        par[i] = np.array(par[i])
    res = {}
    len_parameters = len(par)
    split_num = par[0]
    pre_num = int(len(par[1]) / split_num + 1)
    for i in range(split_num):
        start = i*pre_num
        end = (i+1)*pre_num
        if end > len(par[1]):
            end = len(par[1])
        cur_batch_idxs = list(range(start, end))
        for j in range(1, len_parameters):
            print(j)
            if (j-1) not in res.keys():
                res[(j-1)] = []
                res[(j-1)].append(par[j][cur_batch_idxs])
            else:
                res[(j-1)].append(par[j][cur_batch_idxs])
    final_res = [res[i] for i in res.keys()]
    return final_res


def calculate_acc_error(logits, label, show=True):
    error_index = []
    error_dict = {}
    error_dict_record = {}
    error_count = 0
    error_record = []
    label = np.array(label).squeeze()
    logits = np.array(logits).squeeze()
    for index, logit in enumerate(logits):
        if logit != label[index]:
            error_count += 1
            if label[index] in error_dict.keys():
                error_dict[label[index]] += 1   # 该类别分类错误的个数加１
                error_dict_record[label[index]].append(logit)   # 记录错误的结果
            else:
                error_dict[label[index]] = 1
                error_dict_record[label[index]] = [logit]
            error_index.append(index)
            error_record.append(logit)
    acc = (1.0 * error_count) / (1.0 * len(label))
    if show:
        for key in error_dict.keys():
            print('label is %d, error number is %d, all number is %d, acc is %g'\
                  % (key, error_dict[key], np.sum(label == key), 1-(error_dict[key]*1.0)/(np.sum(label == key) * 1.0)))
            print('error record　is ', error_dict_record[key])
    return error_dict, error_dict_record, acc, error_index, error_record
if __name__ == '__main__':
    image = np.random.random([1000, 40, 40])
    patches = extract_patches_from_images(image, patch_size=5, patch_step=1)
    print(
        np.shape(patches)
    )
