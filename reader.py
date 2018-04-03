import numpy as np
import scipy.io as scio
from tools import show_image
import os

class Reader:
    def __init__(self, train_dir, test_path):
        self.test_images, self.test_labels = Reader.analysis_singlefile(test_path)
        self.train_images, self.train_labels = Reader.analysis_dir(train_dir)
        print('loading training image shape is ', np.shape(self.train_images), ' training label shape is ',
              np.shape(self.train_labels))
        print('loading testing image shape is ', np.shape(self.test_images), ' testing label shape is ',
              np.shape(self.test_labels))
        show_image(self.test_images[0], [100, 100])

    @staticmethod
    def analysis_singlefile(test_path):
        images = []
        labels = []
        data_mat = Reader.read_file(test_path)
        img_shape, img_len = np.shape(data_mat['image'])
        for i in range(0, img_len):
            images.append(np.reshape(data_mat['image'][:, i], [40, 40]))
            labels.append(data_mat['label_int'][i])
        return images, labels
    @staticmethod
    def read_file(file_path):
        data_mat = scio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        return Reader._check_keys(data_mat)['affNISTdata']
    @staticmethod
    def analysis_dir(train_dir):
        names = os.listdir(train_dir)
        images = []
        labels = []
        for name in names:
            image, label = Reader.analysis_singlefile(os.path.join(train_dir, name))
            images.extend(image)
            labels.extend(label)
        return images, labels
    @staticmethod
    def _check_keys(dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], scio.matlab.mio5_params.mat_struct):
                dict[key] = Reader._todict(dict[key])
        return dict

    @staticmethod
    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scio.matlab.mio5_params.mat_struct):
                dict[strg] = Reader._todict(elem)
            else:
                dict[strg] = elem
        return dict
    
if __name__ == '__main__':
    data_dict = Reader.read_file('/home/give/homework/CV/dataset/affNIST/test.mat')
    print(data_dict.keys())
    reader = Reader('/home/give/homework/CV/dataset/affNIST/training_and_validation_batches',
                    '/home/give/homework/CV/dataset/affNIST/test.mat')