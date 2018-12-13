"""
Loading and augmenting the Omniglot dataset.

To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os
import random

from PIL import Image
import numpy as np

def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.

    Args:
      data_dir: a directory of alphabet directories.

    Returns:
      An iterable over Characters.

    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), 0)

def split_dataset(dataset, num_train=1200):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

# pylint: disable=R0903
class Character:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0, blur=False ):
        self.dir_path = dir_path
        self.rotation = rotation
        self.blur = blur
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
            return self._cache[path]
