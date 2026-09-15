"""A uniform eight-token model must report perplexity eight."""

from pathlib import Path
import sys
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import compute_perplexity


class EvaluationTest(unittest.TestCase):
    def test_uniform_predictions_with_padding_and_short_examples(self):
        backend = Tokenizer(WordLevel({f"t{i}": i for i in range(8)}, unk_token="t0"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="t0", unk_token="t0")
        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_embd=8, n_layer=1, n_head=2))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        data = [{"text": text} for text in ("t1 t2 t3", "t4 t5", "t6")]
        for batch_size in (1, 3):
            result = compute_perplexity(model, tokenizer, data, 8, batch_size=batch_size, device="cpu")
            self.assertEqual(len(result["per_example_perplexity"]), 2)
            for value in result["per_example_perplexity"]:
                self.assertAlmostEqual(value, 8, places=5)
            self.assertAlmostEqual(result["mean_perplexity"], 8, places=5)


if __name__ == "__main__":
    unittest.main()
