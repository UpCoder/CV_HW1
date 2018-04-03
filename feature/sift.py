import cv2
import numpy as np
import os, sys

def get_sift(image_arr):
    image_arr = np.squeeze(image_arr)
    if len(np.shape(image_arr)) == 2:
        gray_img = image_arr
    else:
        gray_img = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()


if __name__ == '__main__':
    get_sift(np.zeros([10,10]))
