import unittest
from coregpt import load_dataset, CharTokenizer, TinyLanguageModel
from config import Config


class TestModel(unittest.TestCase):

    def test_forward_shape(self):
        text = load_dataset(Config.dataset_path)
        tok = CharTokenizer(text)
        model = TinyLanguageModel(tok.vocab_size)

        tokens = tok.encode("CoreGPT example text")
        tokens = tokens[:Config.block_size]

        logits = model.forward(tokens)

        self.assertEqual(len(logits[0]), tok.vocab_size)
        self.assertEqual(len(logits), len(tokens))


if __name__ == "__main__":
    unittest.main()