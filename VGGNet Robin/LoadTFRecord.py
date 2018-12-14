import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np
import operator
import os

IN_DIR = 'C:/Users/rthun/BA/Data/TFRecords/'
train_filenames = ['train_128_0-3.tfrecord']
test_filenames = ['test_128_0-3.tfrecord']
val_filenames = ['validation_128_0-3.tfrecord']
DATA_FRAME = pd.read_csv('D://Medien/Dokumente/Bachelor-Arbeit/train.csv')
MOST_FREQUENT_CLASSES = pd.read_csv('C:/Users/rthun/BA/Data/most_frequent_classes.csv')
train_filenames = [IN_DIR + name for name in train_filenames]
test_filenames = [IN_DIR + name for name in test_filenames]
val_filenames = [IN_DIR + name for name in val_filenames]

LABELS = {(x + '.jpg'): y for x, y in zip(DATA_FRAME.id, DATA_FRAME.landmark_id)}


source_img_size = 128
target_img_size = 128
classes_num = 3

def save_most_frequent_classes_to_txt(data, data_path, file_path):
    counts = {x: 0 for x in range(0, 15000)}
    for i, j in zip(data.landmark_id, data.id):
        if data_path is not None:
            if os.path.exists(data_path + j + '.jpg'):
                counts[i] += 1
        else:
            counts[i] += 1

    result = sorted(counts.items(), key=operator.itemgetter(1))
    result.reverse()

    result_1 = [x[0] for x in result]
    result_2 = [x[1] for x in result]
    raw_data = {'id': result_1, 'num': result_2}
    df = pd.DataFrame(raw_data, columns=['id', 'num'])
    df.to_csv(file_path)


def get_filenames_for_class(path, cl):
    names = [str(x + '.jpg') for x in DATA_FRAME.id if str(LABELS[str(x + '.jpg')]) == str(cl)]
    return [(path + name) for name in names if os.path.exists(path + name)]


def get_most_frequent_classes(classes, max_size=None):
    result = [[x, y] for x, y, _ in zip(MOST_FREQUENT_CLASSES.id, MOST_FREQUENT_CLASSES.num, range(classes))]
    #print(result)

    summation = 0
    if max_size is not None:
        size = []
        for x in result:
            if x[1] <= max_size:
                size.append(x[1])
            else:
                size.append(max_size)
        summation = sum(size)
        print(summation)

    result = [str(x[0]) for x in result]
    return result, summation


def calculate_most_frequent_classes(data, classes, data_path=None, max_size=None):
    counts = {x: 0 for x in range(0, 15000)}
    for i, j in zip(data.landmark_id, data.id):
        if data_path is not None:
            if os.path.exists(data_path + j + '.jpg'):
                counts[i] += 1
        else:
            counts[i] += 1

    result = sorted(counts.items(), key=operator.itemgetter(1))
    result.reverse()
    result = result[0:classes]

    summation = 0
    if max_size is not None:
        size = []
        for x in result:
            if x[1] <= max_size:
                size.append(x[1])
            else:
                size.append(max_size)
        summation = sum(size)
        print(summation)

    result = [str(x[0]) for x in result]
    return result, summation

#classes = calculate_most_frequent_classes(DATA_FRAME, 100, data_path='C:/Users/rthun/BA/Data/128/', max_size=1000)
#save_most_frequent_classes_to_txt(data=DATA_FRAME, data_path='C:/Users/rthun/BA/Data/128/', file_path='C:/Users/rthun/BA/Data/most_frequent_classes.csv')


def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    image = tf.multiply(image, 1.0 / 255.0)

    # Get the label associated with the image.
    label = parsed_example['label']


    # The image and label are now correct TensorFlow types.
    return image, label


def input_fn(filenames, train, batch_size=64, buffer_size=2048):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse, num_parallel_calls=1)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(buffer_size=1)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = images_batch
    y = labels_batch

    return x, y


def train_input_fn():
    return input_fn(filenames=train_filenames, train=True)


def test_input_fn():
    return input_fn(filenames=test_filenames, train=False)