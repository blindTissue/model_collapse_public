import unittest
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pipeline import commands


class PipelineTest(unittest.TestCase):
    def test_pool_barrier_and_chained_training(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/fullrank.yaml").read_text())
        steps = list(commands(config, root / "configs/fullrank.yaml", Path("/run"), 0, 1, 42))
        names = [step[0][0] for step in steps]
        expected = ["finetune_fullrank.py"] * 3 + ["generate_data.py"] * 3 + ["pool_synthetic.py"] + ["evaluate.py"] * 3
        self.assertEqual(names, ["prepare_data.py"] + expected * 2)
        first_second_gen = steps[11][0]
        self.assertIn("/run/models/llama/gen_0", first_second_gen)
        self.assertIn("/run/data/synthetic_gen_0/mixed", first_second_gen)

    def test_qlora_training_seed_is_explicit(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/qlora.yaml").read_text())
        steps = list(commands(config, root / "configs/qlora.yaml", Path("/run"), 0, 0, 123))
        train = steps[1][0]
        self.assertEqual(train[train.index("--seed") + 1], "123")
        self.assertEqual(train[0], "finetune_qlora.py")


if __name__ == "__main__":
    unittest.main()
