"""Regression checks for process isolation, checkpoint identity and resume."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "xm_cz" / "runner.py"
spec = importlib.util.spec_from_file_location("parallel_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ParallelRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "output").mkdir(exist_ok=True)

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.make_tasks({"tasks": [{"x": 1}, {"x": 1}]})

    def test_completed_grid_extension_reuses_points(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            directory = Path(temporary)
            path = directory / "config.json"
            config = {
                "schema_version": 1,
                "workers": 2,
                "handler": "xm_cz.diagnostics:diagnostic_task",
                "output_directory": "results",
                "tasks": [{"values": [1]}],
            }
            path.write_text(json.dumps(config))
            command = [sys.executable, str(RUNNER), str(path)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=40)
            self.assertEqual(first.returncode, 0, first.stderr)
            checkpoint = next((directory / "results/points").glob("*.json"))
            timestamp = checkpoint.stat().st_mtime_ns
            config.update(extend_completed_run=True)
            config["tasks"].append({"values": [2]})
            path.write_text(json.dumps(config))
            extended = subprocess.run(
                command, capture_output=True, text=True, timeout=40
            )
            self.assertEqual(extended.returncode, 0, extended.stderr)
            self.assertIn("1 cached; 1 pending", extended.stdout)
            self.assertEqual(checkpoint.stat().st_mtime_ns, timestamp)
            self.assertEqual(
                len(list((directory / "results/points").glob("*.json"))), 2
            )

    def test_timeout_checkpoints_and_resumes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            directory = Path(temporary)
            path = directory / "config.json"
            config = {
                "schema_version": 1,
                "workers": 4,
                "handler": "xm_cz.diagnostics:diagnostic_task",
                "output_directory": "results",
                "task_timeout_seconds": 0.00001,
                "tasks": [{"values": [i, 1]} for i in range(4)],
            }
            path.write_text(json.dumps(config))
            command = [sys.executable, str(RUNNER), str(path)]
            interrupted = subprocess.run(
                command, capture_output=True, text=True, timeout=40
            )
            self.assertEqual(interrupted.returncode, 2)
            manifest = json.loads((directory / "results/manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            config["task_timeout_seconds"] = 30
            path.write_text(json.dumps(config))
            resumed = subprocess.run(
                command, capture_output=True, text=True, timeout=40
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            manifest = json.loads((directory / "results/manifest.json").read_text())
            self.assertEqual(manifest["completed"], 4)
            self.assertEqual(manifest["status"], "complete")

    def test_resume_failure_capture_and_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            directory = Path(temporary)
            config_path = directory / "config.json"
            config = {
                "schema_version": 1,
                "workers": 4,
                "handler": "xm_cz.diagnostics:diagnostic_task",
                "output_directory": "results",
                "tasks": [{"values": [1, 2, 3]}, {"raise_error": True}],
            }
            config_path.write_text(json.dumps(config))

            def invoke(*extra):
                return subprocess.run(
                    [sys.executable, str(RUNNER), str(config_path), *extra],
                    capture_output=True,
                    text=True,
                    timeout=40,
                )

            first = invoke()
            self.assertEqual(first.returncode, 1, first.stderr)
            checkpoints = directory / "results/points"
            records = [json.loads(p.read_text()) for p in checkpoints.glob("*.json")]
            self.assertEqual(sorted(r["status"] for r in records), ["failed", "ok"])
            self.assertEqual(
                next(r for r in records if r["status"] == "ok")["result"]["sum"], 6
            )
            self.assertTrue(
                all(all(n == 1 for n in r["blas_threads"]) for r in records)
            )
            timestamps = {
                p.name: p.stat().st_mtime_ns for p in checkpoints.glob("*.json")
            }
            second = invoke("--workers", "1")
            self.assertEqual(second.returncode, 1, second.stderr)
            self.assertIn("0 pending", second.stdout)
            self.assertEqual(
                timestamps,
                {p.name: p.stat().st_mtime_ns for p in checkpoints.glob("*.json")},
            )
            retry = invoke("--retry-failed")
            self.assertEqual(retry.returncode, 1, retry.stderr)
            self.assertIn("1 pending", retry.stdout)
            config["tasks"][0]["values"] = [9]
            config_path.write_text(json.dumps(config))
            mismatch = invoke()
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("different configuration", mismatch.stderr)


if __name__ == "__main__":
    (ROOT / "output").mkdir(exist_ok=True)
    unittest.main()
