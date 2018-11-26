import unittest
import supervised_reptile.cuneiform as cuneiform
from PIL import Image

class TestCuneiform(unittest.TestCase):

    def test_read_data(self):
            data = cuneiform.read_dataset('../data/cuneiform')
            for i in  data:
                img = i.sample(1)
                test = 0