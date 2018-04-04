import numpy as np
import skimage.feature as skft
from reader import Reader
from classification import LinearSVM
import scipy.io as scio
from tools import calculate_acc_error
import time


class Config:
    n_points = 8
    radius = 3


def get_LBP_features(train_images, train_labels, test_images, test_labels):
    train_features = []
    train_labels_res = []
    for idx, train_image in enumerate(train_images):
        lbp = skft.local_binary_pattern(train_image, Config.n_points, Config.radius, 'uniform')
        max_bins = int(lbp.max() + 1)
        cur_feature, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
        if len(cur_feature) != 10:
            continue
        if idx % 10000 == 0:
            print('%d / %d' % (idx, len(train_images)))
        train_features.append(cur_feature)
        train_labels_res.append(train_labels[idx])
    test_features = []
    test_labels_res = []
    for idx, test_image in enumerate(test_images):
        lbp = skft.local_binary_pattern(test_image, Config.n_points, Config.radius, 'uniform')
        max_bins = int(lbp.max() + 1)
        cur_feature, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
        if len(cur_feature) != 10:
            continue
        if idx % 10000 == 0:
            print('%d / %d' % (idx, len(test_images)))
        test_features.append(cur_feature)
        test_labels_res.append(test_labels[idx])
    return train_features, train_labels_res, test_features, test_labels_res


if __name__ == '__main__':
    reader = Reader(
        '/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
        '/home/give/homework/CV/dataset/affNIST/test.mat'
    )
    time1 = time.time()
    train_features, train_labels, test_features, test_labels = get_LBP_features(reader.train_images,
                                                                                reader.train_labels, reader.test_images,
                                                                                reader.test_labels)

    print('LBP train features shape is ', np.shape(train_features))
    print('LBP test features shape is ', np.shape(test_features))
    time2 = time.time()
    print('Extract LBP feature cost ', (time2 - time1))
    scio.savemat('./LBP_features.mat', {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels
    })
    predicted = LinearSVM.do(train_data=train_features, train_label=train_labels, test_data=test_features,
                             test_label=test_labels)
    time3 = time.time()
    print('SVM cost ', (time3 - time2))
    np.save('./LBP_predicted.npy', predicted)
    calculate_acc_error(predicted, test_labels)