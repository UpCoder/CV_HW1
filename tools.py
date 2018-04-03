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
if __name__ == '__main__':
    image = np.random.random([1000, 40, 40])
    patches = extract_patches_from_images(image, patch_size=5, patch_step=1)
    print(
        np.shape(patches)
    )
