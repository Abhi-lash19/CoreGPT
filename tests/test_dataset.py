import unittest
from coregpt import load_dataset
from config import Config


class TestDataset(unittest.TestCase):

    def test_dataset_load(self):
        text = load_dataset(Config.dataset_path)
        self.assertTrue(len(text) > 0)


if __name__ == "__main__":
    unittest.main()