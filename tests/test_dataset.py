import unittest
from coregpt import load_dataset, CharTokenizer
from config import Config


class TestDataset(unittest.TestCase):

    def test_dataset_load(self):
        text = load_dataset(Config.dataset_path)
        self.assertTrue(len(text) > 0)

    def test_tokenizer(self):
        text = load_dataset(Config.dataset_path)
        tok = CharTokenizer(text)

        encoded = tok.encode("test")
        decoded = tok.decode(encoded)

        self.assertEqual(decoded, "test")


if __name__ == "__main__":
    unittest.main()