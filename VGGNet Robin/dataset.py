import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import pandas as pd


IN_DIR_TRAIN = 'C:/Users/rthun/BA/Data/100Classes128/training/'
IN_DIR_TEST = 'C:/Users/rthun/BA/Data/100Classes128/test/'
DATA_FRAME = pd.read_csv('D://Medien/Dokumente/Bachelor-Arbeit/train.csv')
LABELS = {(x + '.jpg'): y for x, y in zip(DATA_FRAME.id, DATA_FRAME.landmark_id)}


def get_filenames(path):
    names = [str(x + '.jpg') for x in DATA_FRAME.id]
    return [(path + name) for name in names if os.path.exists(path + name)]


def get_filenames_for_class(path, cl, max_size):
    names = [str(x + '.jpg') for x in DATA_FRAME.id if str(LABELS[str(x + '.jpg')]) == str(cl)]
    result = []
    complete = False
    names_iterator = iter(names)
    while (len(result) < max_size) and not complete:
        try:
            name = next(names_iterator)
            if os.path.exists(path + name):
                result.append(path + name)
        except StopIteration:
            complete = True
    return result


def write_filename(name, filename):
    with open(filename, "a") as text_file:
        text_file.write(name + '\n')


def load_train(train_path, image_size, classes, max_size):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        files = get_filenames_for_class(train_path, fields, max_size)
        print('Now going to read {} files (Index: {})'.format(fields, index), 'with length:', len(files))
        read_size = 0
        for fl in files:
            try:
                image = cv2.imread(fl)
                if(image_size != 0):
                    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                img_names.append(flbase)
                cls.append(fields)
                read_size += 1
                if read_size % 2500 == 0:
                    print('Read:', read_size)
            except:
                print('Can not read image:', fl)

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    def shuffle(self):
        self._images, self._labels, self._img_names, self._cls = shuffle(self._images, self._labels, self._img_names, self._cls)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch >= self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            self._index_in_epoch = 0
            assert batch_size <= self._num_examples
            end = self._num_examples
            return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]
        else:
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, max_size, validation_portion, test_path=None):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes, max_size)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    validation_size = 0

    if (test_path is not None) and (isinstance(validation_portion, float)):
        validation_images, validation_labels, validation_img_names, validation_cls = load_train(test_path, image_size, classes, int((max_size * validation_portion) / (1 - validation_portion)))
        validation_images, validation_labels, validation_img_names, validation_cls = shuffle(validation_images, validation_labels, validation_img_names, validation_cls)
    else:
        if isinstance(validation_portion, float):
            validation_size = int(validation_portion * images.shape[0])
            print(validation_size)

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets


