#!/usr/bin/env python3
"""End-to-end checks for the built SGEMM executable.

The test uses only Python's standard library. GPU-dependent checks return 77
when CUDA reports that no device or usable driver is available; all other
execution and output failures are real test failures.
"""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = None
SKIP = 77
GPU_UNAVAILABLE = re.compile(
    r"no CUDA-capable device|no CUDA device|insufficient driver|"
    r"outside the available CUDA device",
    re.IGNORECASE,
)


class CliEndToEndTests(unittest.TestCase):
    def run_cli(self, *arguments, gpu=True):
        environment = os.environ.copy()
        if gpu:
            # A parent shell may have hidden the device for a host-only check.
            environment.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            [str(EXECUTABLE), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=120,
        )
        return result

    def assert_success(self, result, context):
        self.assertEqual(
            result.returncode,
            0,
            f"{context}: exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_host_listing_does_not_initialize_cuda(self):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [str(EXECUTABLE), "--list-kernels"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=30,
        )
        self.assert_success(result, "--list-kernels")
        self.assertEqual(result.stderr, "")
        for expected in ("0  cublas-fp32", "10  warp-tiled", "12  async-double-buffered", "auto"):
            self.assertIn(expected, result.stdout)

    def require_gpu(self):
        # The module-level probe in main() converts unavailable hardware into
        # CTest's SKIP_RETURN_CODE=77 before unittest starts.
        self.assertIsNotNone(EXECUTABLE)

    @staticmethod
    def read_one_csv(path):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise AssertionError(f"expected one CSV row, got {len(rows)}")
        return rows[0]

    def test_auto_irregular_falls_back_to_cublas(self):
        self.require_gpu()
        with tempfile.TemporaryDirectory(prefix="sgemm-cli-e2e-") as directory:
            result_path = Path(directory) / "irregular.csv"
            result = self.run_cli(
                "--kernel", "auto", "--m", "7", "--n", "13", "--k", "5",
                "--warmup", "0", "--iters", "1", "--csv", str(result_path),
            )
            self.assert_success(result, "auto irregular workload")
            row = self.read_one_csv(result_path)
            self.assertEqual(row["kernel"], "0")
            self.assertEqual(row["kernel_name"], "cublas-fp32")
            self.assertEqual(row["requested_kernel"], "auto")
            self.assertEqual((row["size"], row["m"], row["n"], row["k"]),
                             ("", "7", "13", "5"))
            self.assertIn("not measured/autotuned", row["selection_reason"])
            self.assertGreater(float(row["average_seconds"]), 0.0)
            # CSV GFLOPS is formatted to three decimals; tiny workloads can
            # legitimately round below 0.001 GFLOP/s.
            self.assertGreaterEqual(float(row["gflops"]), 0.0)
            self.assertEqual(row["verified"], "false")
            self.assertIn("selected kernel 0 (cublas-fp32)", result.stdout)

    def test_named_naive_rectangular_workload(self):
        self.require_gpu()
        with tempfile.TemporaryDirectory(prefix="sgemm-cli-e2e-") as directory:
            result_path = Path(directory) / "naive.csv"
            result = self.run_cli(
                "--kernel", "naive", "--m", "7", "--n", "13", "--k", "5",
                "--warmup", "0", "--iters", "1", "--csv", str(result_path),
            )
            self.assert_success(result, "named naive workload")
            row = self.read_one_csv(result_path)
            self.assertEqual(row["kernel"], "1")
            self.assertEqual(row["kernel_name"], "naive")
            self.assertEqual(row["requested_kernel"], "naive")
            self.assertEqual((row["size"], row["m"], row["n"], row["k"]),
                             ("", "7", "13", "5"))
            self.assertEqual(row["verified"], "true")
            self.assertIn("dimensions: (m=7, n=13, k=5)", result.stdout)

    def test_auto_aligned_workload_reports_policy_selection(self):
        self.require_gpu()
        with tempfile.TemporaryDirectory(prefix="sgemm-cli-e2e-") as directory:
            result_path = Path(directory) / "aligned.csv"
            result = self.run_cli(
                "--kernel", "auto", "--m", "128", "--n", "256", "--k", "32",
                "--warmup", "0", "--iters", "1", "--csv", str(result_path),
            )
            self.assert_success(result, "auto aligned workload")
            row = self.read_one_csv(result_path)
            self.assertEqual(row["requested_kernel"], "auto")
            self.assertEqual((row["m"], row["n"], row["k"]), ("128", "256", "32"))
            self.assertEqual(row["size"], "")
            self.assertEqual(row["kernel_name"], "warp-tiled")
            self.assertEqual(row["kernel"], "10")
            self.assertIn("policy", row["selection_reason"])
            self.assertNotIn("fastest", row["selection_reason"])
            self.assertEqual(row["verified"], "true")

    def test_explicit_unsupported_shape_fails_preflight(self):
        self.require_gpu()
        result = self.run_cli(
            "--kernel", "warp-tiled", "--m", "7", "--n", "13", "--k", "5",
            "--warmup", "0", "--iters", "1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported", result.stderr.lower())
        self.assertIn("require", result.stderr.lower())
        self.assertNotIn("Average elapsed time", result.stdout)

    def test_incomplete_shape_is_rejected_without_gpu(self):
        result = self.run_cli("--kernel", "auto", "--m", "128", gpu=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--m, --n, and --k must all be specified together", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    arguments = parser.parse_args()
    EXECUTABLE = arguments.executable.resolve()
    if not EXECUTABLE.is_file():
        parser.error(f"executable does not exist: {EXECUTABLE}")

    # Probe outside unittest so an unavailable GPU produces the documented
    # process-level skip code rather than a misleading all-tests-passed result.
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    probe = subprocess.run(
        [str(EXECUTABLE), "--kernel", "auto", "--m", "7", "--n", "13", "--k", "5",
         "--warmup", "0", "--iters", "1"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, timeout=120,
    )
    if probe.returncode != 0:
        if GPU_UNAVAILABLE.search(probe.stderr):
            print("SKIP: CUDA device/driver unavailable", file=sys.stderr)
            raise SystemExit(SKIP)
        print(probe.stdout, end="", file=sys.stderr)
        print(probe.stderr, end="", file=sys.stderr)
        raise SystemExit(probe.returncode)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
