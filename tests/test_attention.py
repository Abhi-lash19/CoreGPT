import unittest
from coregpt import load_dataset, CharTokenizer, TinyLanguageModel
from config import Config


class TestAttention(unittest.TestCase):

    def test_attention_forward(self):
        text = load_dataset(Config.dataset_path)
        tok = CharTokenizer(text)
        model = TinyLanguageModel(tok.vocab_size)

        tokens = tok.encode("CoreGPT attention test")
        tokens = tokens[:Config.block_size]

        logits = model.forward(tokens)

        self.assertEqual(len(logits[0]), tok.vocab_size)
        self.assertEqual(len(logits), len(tokens))


if __name__ == "__main__":
    unittest.main()