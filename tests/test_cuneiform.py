import unittest
import supervised_reptile.cuneiform as cuneiform

class TestCuneiform(unittest.TestCase):

    def test_read_data(self):
            data = cuneiform.read_dataset('../data/cuneiform')
            self.assertEqual(data[0], )