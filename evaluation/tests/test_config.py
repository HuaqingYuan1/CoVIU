import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shared_eval_runner", ROOT / "run_eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ConfigurationTests(unittest.TestCase):
    def resolve(self, *arguments):
        return runner.resolve_arguments(runner.build_parser().parse_args(arguments))

    def test_profiles_resolve_from_another_working_directory_without_gpu(self):
        for method in ("coviu", "ga", "gadiff", "npo", "manifold", "viker", "base"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "run_eval.py"), "--method", method, "--dry-run"],
                    cwd=cwd, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                cfg = json.loads(result.stdout)
                self.assertEqual(cfg["method"], method)
                self.assertEqual(cfg["few_shots_num"], 0)
                self.assertTrue(Path(cfg["arguments"]["cache_path"]).is_absolute())
                self.assertEqual(cfg["arguments"]["adapter_path"] is None, method == "base")

    def test_overrides_keep_adapter_and_ratio_explicit(self):
        args, _ = self.resolve("--method", "ga_diff", "--adapter-path", "../baseline/saved_model/forget5_ga_diff", "--forget-ratio", "5", "--seed", "44")
        self.assertEqual(args.forget_ratio, 5)
        self.assertEqual(args.seed, 44)
        self.assertEqual(Path(args.adapter_path), (ROOT.parent / "baseline/saved_model/forget5_ga_diff").resolve())
        self.assertEqual(args.output_file, "gadiff_forget5_seed44")

    def test_invalid_ratio_and_unsafe_output_label(self):
        for arguments in (("--forget-ratio", "0"), ("--forget-ratio", "100"), ("--name", "../escape")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.resolve(*arguments)

    def test_fixed_split_order(self):
        args, _ = self.resolve("--splits", "real", "test", "forget")
        self.assertEqual(args.eval_list, ["forget", "test", "real"])

    def test_base_only_requires_selected_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "base").mkdir()
            (path / "processor").mkdir()
            (path / "real.parquet").touch()
            args, _ = self.resolve("--method", "base", "--splits", "real")
            args.cache_path = str(path / "base")
            args.processor_id = str(path / "processor")
            args.celebrity_data = str(path / "real.parquet")
            runner.validate_inputs(args)
            (path / "real.parquet").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Parquet"):
                runner.validate_inputs(args)

    def test_adapter_config_without_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = Path(folder)
            (adapter / "adapter_config.json").write_text("{}")
            args, _ = self.resolve()
            args.adapter_path = str(adapter)
            with self.assertRaisesRegex(FileNotFoundError, "adapter weights"):
                runner.validate_inputs(args)

    def test_test_split_requires_forget_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "test.parquet").touch()
            args, _ = self.resolve("--method", "base", "--splits", "test")
            args.cache_path = args.processor_id = str(path)
            args.data_split_folder = str(path)
            args.test_data = str(path / "test.parquet")
            with self.assertRaisesRegex(FileNotFoundError, "forget_5"):
                runner.validate_inputs(args)


if __name__ == "__main__":
    unittest.main()
