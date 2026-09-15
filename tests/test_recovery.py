"""Exercise both recovery operators using tiny, locally constructed models."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recover_qlora import build_recovered


class RecoveryTest(unittest.TestCase):
    def make_model(self, destination):
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_embd=8, n_layer=1, n_head=2,
                                         bos_token_id=1, eos_token_id=2, pad_token_id=0))
        model.save_pretrained(destination)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(8)}, unk_token="t0")),
            unk_token="t0", bos_token="t1", eos_token="t2", pad_token="t0")
        tokenizer.save_pretrained(destination)
        return model

    def test_fullrank_cli_negates_from_current_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            current_path = base / "models/tiny/gen_0"
            current = self.make_model(current_path)
            original = {k: v.clone() for k, v in current.state_dict().items()}
            with torch.no_grad():
                for parameter in current.parameters():
                    parameter.add_(0.03)
            future = {k: v.clone() for k, v in current.state_dict().items()}
            current.save_pretrained(base / "models/tiny/gen_1")
            config = base / "config.yaml"
            config.write_text("model:\n  dtype: float32\npaths:\n  models_dir: models\n")
            subprocess.run([sys.executable, str(ROOT / "src/recover_fullrank.py"),
                            "--config", str(config), "--base_dir", str(base), "--family", "tiny",
                            "--current_gen", "0", "--future_gen", "1", "--out_gen", "2", "--alpha", "2"],
                           check=True, capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            recovered = AutoModelForCausalLM.from_pretrained(base / "models/tiny/gen_2")
            preserved = AutoModelForCausalLM.from_pretrained(current_path)
            for name, actual in recovered.state_dict().items():
                torch.testing.assert_close(actual, original[name] - 2 * (future[name] - original[name]))
                torch.testing.assert_close(preserved.state_dict()[name], original[name])

    def test_lora_sign_and_coefficient(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model = self.make_model(base / "current")
            weight = model.transformer.h[0].attn.c_attn.weight.detach().clone()
            peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"],
                                                  task_type="CAUSAL_LM", fan_in_fan_out=True))
            layer = peft.base_model.model.transformer.h[0].attn.c_attn
            with torch.no_grad():
                layer.lora_A["default"].weight.fill_(0.02)
                layer.lora_B["default"].weight.fill_(0.03)
            delta = (layer.lora_B["default"].weight @ layer.lora_A["default"].weight).T.detach()
            peft.save_pretrained(base / "adapter")
            build_recovered(str(base / "current"), str(base / "adapter"), 1.25,
                            str(base / "recovered"), torch.float32, "eager")
            recovered = AutoModelForCausalLM.from_pretrained(base / "recovered")
            torch.testing.assert_close(recovered.transformer.h[0].attn.c_attn.weight, weight - 1.25 * delta)


if __name__ == "__main__":
    unittest.main()
