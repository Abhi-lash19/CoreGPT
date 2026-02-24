import unittest
from coregpt import load_dataset, CharTokenizer, TinyLanguageModel, train_step
from config import Config


class TestTraining(unittest.TestCase):

    def test_training_runs(self):
        text = load_dataset(Config.dataset_path)
        tok = CharTokenizer(text)
        model = TinyLanguageModel(tok.vocab_size)

        tokens = tok.encode("CoreGPT training step")
        tokens = tokens[:Config.block_size]

        loss = train_step(model, tokens, tokens)
        self.assertTrue(loss > 0)


if __name__ == "__main__":
    unittest.main()