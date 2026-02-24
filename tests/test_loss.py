import unittest
from coregpt import load_dataset, CharTokenizer, TinyLanguageModel, cross_entropy_loss
from config import Config


class TestLoss(unittest.TestCase):

    def test_loss_positive(self):
        text = load_dataset(Config.dataset_path)
        tok = CharTokenizer(text)
        model = TinyLanguageModel(tok.vocab_size)

        tokens = tok.encode("CoreGPT learning test")
        tokens = tokens[:Config.block_size]

        logits = model.forward(tokens)
        loss = cross_entropy_loss(logits, tokens)

        self.assertTrue(loss > 0)


if __name__ == "__main__":
    unittest.main()