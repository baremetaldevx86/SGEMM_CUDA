#!/usr/bin/env python3
"""CPU-only regression tests for the real sgemm.cu compiled with runner.cuh.

Usage: python3 tests/benchmark_mock/test_benchmark.py /path/to/sgemm_mock
Only Python's standard library is used. Every subprocess gets an isolated cwd.
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
SIZES = [128, 256, 512, 1024, 2048, 4096]
FIELDS = ["kernel", "size", "average_seconds", "gflops", "warmup", "iters",
          "seed", "verified", "alpha", "beta"]
FAST = ["1", "--warmup", "0", "--iters", "1"]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sgemm-benchmark-mock-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case_number = 0

    def run_benchmark(self, arguments, expected=0, environment=None, setup=None):
        self.case_number += 1
        cwd = self.root / str(self.case_number)
        cwd.mkdir()
        if setup is not None:
            setup(cwd)
        env = {key: value for key, value in os.environ.items()
               if key != "DEVICE" and not key.startswith("SGEMM_MOCK_")}
        env.update(environment or {})
        result = subprocess.run([str(EXECUTABLE), *arguments], cwd=str(cwd),
                                env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(result.returncode, expected,
                         "arguments={!r}, environment={!r}\nstdout:\n{}\nstderr:\n{}"
                         .format(arguments, environment, result.stdout, result.stderr))
        return result, cwd

    def read_csv(self, path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, FIELDS)
            return list(reader)

    def check_success(self, result, cwd, kernel, warmup=5, iterations=50,
                      alpha=0.5, beta=3.0, seed=1234, supported=None):
        supported = SIZES if supported is None else supported
        rows = self.read_csv(cwd / "results.csv")
        self.assertEqual([int(row["size"]) for row in rows], supported)
        average_seconds = (iterations + 1) / 2000.0
        for row in rows:
            self.assertEqual(int(row["kernel"]), kernel)
            self.assertEqual(int(row["warmup"]), warmup)
            self.assertEqual(int(row["iters"]), iterations)
            self.assertEqual(int(row["seed"]), seed)
            self.assertEqual(float(row["alpha"]), alpha)
            self.assertEqual(float(row["beta"]), beta)
            self.assertAlmostEqual(float(row["average_seconds"]), average_seconds)
            gflops = 2 * int(row["size"]) ** 3 * 1e-9 / average_seconds
            self.assertAlmostEqual(float(row["gflops"]), gflops, delta=0.000501)
            self.assertEqual(row["verified"], "true" if kernel else "false")

        stats = re.findall(r"MOCK_SIZE: size=(\d+) supported=(\d+) launches=(\d+) "
                           r"resets=(\d+) timed=(\d+) verifications=(\d+)", result.stdout)
        self.assertEqual([int(stat[0]) for stat in stats], SIZES)
        per_size = warmup + iterations + (2 if kernel else 0)
        for size, available, launches, resets, timed, verifications in stats:
            active = int(size) in supported
            expected_launches = per_size if active else 0
            self.assertEqual(int(available), int(active))
            self.assertEqual(int(launches), expected_launches)
            self.assertEqual(int(resets), expected_launches)
            self.assertEqual(int(timed), iterations if active else 0)
            self.assertEqual(int(verifications), int(active and kernel != 0))
        total = per_size * len(supported)
        self.assertIn("MOCK_CHECK: launches={} resets={} timed={} transitions=6"
                      .format(total, total, iterations * len(supported)), result.stdout)
        self.assertIn("reset copies are excluded from GEMM timing", result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertFalse((cwd / "matrixValidationFailure.txt").exists())

    def test_lifecycle_scalars_csv_and_all_size_transitions(self):
        cases = [
            # The default beta=3 and 50 iterations expose accumulated-C state.
            (["1"], dict(kernel=1)),
            (["--kernel", "0", "--warmup", "2", "--iters", "3"],
             dict(kernel=0, warmup=2, iterations=3)),
            (["12", "--warmup", "0", "--iters", "2", "--alpha", "-0.25",
              "--beta", "-2"],
             dict(kernel=12, warmup=0, iterations=2, alpha=-0.25, beta=-2.0)),
            ([*FAST, "--alpha", "0", "--beta", "0", "--seed", "4294967295"],
             dict(kernel=1, warmup=0, iterations=1, alpha=0.0, beta=0.0,
                  seed=4294967295)),
            (["--kernel", "2", "--warmup", "1", "--iters", "4", "--alpha",
              "2.5e-1", "--beta", "1.25", "--seed", "0"],
             dict(kernel=2, warmup=1, iterations=4, alpha=0.25, beta=1.25, seed=0)),
        ]
        for args, options in cases:
            with self.subTest(args=args):
                result, cwd = self.run_benchmark(
                    [*args, "--csv", "results.csv"], environment={
                        "SGEMM_MOCK_EXPECT_ALPHA": str(options.get("alpha", 0.5)),
                        "SGEMM_MOCK_EXPECT_BETA": str(options.get("beta", 3.0)),
                        "DEVICE": "0",
                    })
                self.check_success(result, cwd, **options)

    def test_kernel11_skips_128_without_launch_or_csv_row(self):
        result, cwd = self.run_benchmark(
            ["11", "--warmup", "0", "--iters", "2", "--csv", "results.csv"])
        self.assertIn("Skipping kernel 11 at size 128: mock preset", result.stdout)
        self.assertEqual(result.stdout.count("Skipping kernel"), 1)
        self.check_success(result, cwd, kernel=11, warmup=0, iterations=2,
                           supported=SIZES[1:])

    def test_all_unsupported_sizes_produce_only_csv_header(self):
        result, cwd = self.run_benchmark(
            [*FAST, "--csv", "results.csv"],
            environment={"SGEMM_MOCK_FAIL": "skip_all"})
        self.assertEqual(result.stdout.count("Skipping kernel"), len(SIZES))
        self.check_success(result, cwd, kernel=1, warmup=0, iterations=1, supported=[])

    def test_csv_is_optional(self):
        result, cwd = self.run_benchmark(FAST)
        self.assertIn("MOCK_CHECK: launches=18 resets=18 timed=6", result.stdout)
        self.assertEqual(list(cwd.iterdir()), [])

    def test_help_requires_no_device(self):
        for flag in ["--help", "-h"]:
            with self.subTest(flag=flag):
                result, _ = self.run_benchmark(
                    [flag], environment={"DEVICE": "invalid", "SGEMM_MOCK_FAIL": "device_count"})
                self.assertIn("Usage:", result.stderr)
                self.assertIn("--alpha", result.stderr)
                self.assertIn("--beta", result.stderr)
                self.assertIn("reset copies are not timed", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_cli_has_useful_diagnostic(self):
        cases = [
            ([], "range 0-12"), (["13"], "range 0-12"),
            (["--kernel", "-1"], "range 0-12"),
            (["1", "--kernel", "2"], "only once"),
            (["--kernel", "1", "2"], "Unknown argument"),
            (["--kernel", "1", "--kernel", "1"], "only once"),
            (["bogus"], "Invalid value for kernel"),
            (["1", "--warmup", "-1"], "--warmup must be non-negative"),
            (["1", "--iters", "0"], "--iters must be greater than zero"),
            (["1", "--iters", "-1"], "--iters must be greater than zero"),
            (["1", "--csv", ""], "--csv requires a non-empty path"),
            (["1", "--bogus"], "Unknown argument"),
        ]
        for option in ["--kernel", "--warmup", "--iters", "--seed", "--alpha", "--beta", "--csv"]:
            prefix = [] if option == "--kernel" else ["1"]
            cases.append(([*prefix, option], "Missing value for " + option))
        for value in ["9999999999999999", "1x", " 1", "1 ", "", "1\n"]:
            cases.append((["1", "--iters", value], "--iters"))
        for value in ["-1", "-18446744073709551615", "4294967296", " 123", "", "12x"]:
            cases.append((["1", "--seed", value], "--seed"))
        for option in ["--alpha", "--beta"]:
            for value in ["nan", "inf", "-inf", "1e100", "3.0x", " 0.5", "", "0.5\t"]:
                cases.append((["1", option, value], option))
        for args, diagnostic in cases:
            with self.subTest(args=args):
                result, _ = self.run_benchmark(args, expected=1)
                self.assertIn("Usage:", result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_device(self):
        for value in ["", "-1", "1", "abc", "0junk", " 0", "0 ", "99999999999999999"]:
            with self.subTest(value=value):
                result, _ = self.run_benchmark(FAST, expected=1, environment={"DEVICE": value})
                self.assertIn("Benchmark failed:", result.stderr)
                self.assertIn("DEVICE", result.stderr)
                self.assertNotIn("MOCK_CHECK:", result.stdout)
        result, _ = self.run_benchmark(FAST, expected=1,
                                       environment={"SGEMM_MOCK_FAIL": "no_devices"})
        self.assertIn("outside the available CUDA device range", result.stderr)

    def test_cuda_and_cublas_failures_are_checked(self):
        failures = {name: "mock CUDA failure: " + name for name in [
            "device_count", "set_device", "event_create", "malloc", "copy", "reset",
            "launch", "device_sync", "event_record", "event_sync", "event_elapsed",
            "event_destroy", "free",
            # Reach later resources, validation downloads, and timed launches,
            # not just the first call of each operation.
            "event_create:2", "malloc:5", "copy:4", "copy:5", "reset:3",
            "launch:3", "event_record:2", "event_destroy:2", "free:5",
        ]}
        failures.update({"cublas_create": "cublasCreate failed with cuBLAS status",
                         "cublas_stream": "cublasSetStream failed with cuBLAS status",
                         "cublas_destroy": "cublasDestroy failed with cuBLAS status"})
        for operation, diagnostic in failures.items():
            with self.subTest(operation=operation):
                result, _ = self.run_benchmark(
                    FAST, expected=1, environment={"SGEMM_MOCK_FAIL": operation})
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn("MOCK invariant failed", result.stderr)

    def test_invalid_elapsed_times_do_not_emit_csv_rows(self):
        for operation in ["zero_time", "negative_time", "nan_time", "infinite_time"]:
            with self.subTest(operation=operation):
                result, cwd = self.run_benchmark(
                    [*FAST, "--csv", "results.csv"], expected=1,
                    environment={"SGEMM_MOCK_FAIL": operation})
                self.assertIn("CUDA events reported an invalid elapsed time", result.stderr)
                self.assertEqual(self.read_csv(cwd / "results.csv"), [])

    def test_validation_failure_logs_original_c_and_no_csv_row(self):
        result, cwd = self.run_benchmark(
            [*FAST, "--csv", "results.csv"], expected=1,
            environment={"SGEMM_MOCK_FAIL": "verify"})
        self.assertIn("Kernel correctness verification failed", result.stderr)
        log = (cwd / "matrixValidationFailure.txt").read_text()
        self.assertIn("Initial C:\n3\n", log)
        self.assertIn("A:\n1\nB:\n2\n", log)
        self.assertIn("Should:\n", log)
        self.assertEqual(self.read_csv(cwd / "results.csv"), [])

    def test_csv_open_failure(self):
        result, _ = self.run_benchmark(
            [*FAST, "--csv", "missing-directory/results.csv"], expected=1)
        self.assertIn("Unable to open CSV output file", result.stderr)

    def test_validation_log_open_failure(self):
        result, _ = self.run_benchmark(
            FAST, expected=1, environment={"SGEMM_MOCK_FAIL": "verify"},
            setup=lambda cwd: (cwd / "matrixValidationFailure.txt").mkdir())
        self.assertIn("Unable to open validation log", result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "write-error device /dev/full is unavailable")
    def test_csv_write_failure(self):
        result, _ = self.run_benchmark([*FAST, "--csv", "/dev/full"], expected=1)
        self.assertIn("Unable to write CSV output file", result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "write-error device /dev/full is unavailable")
    def test_validation_log_write_failure(self):
        result, _ = self.run_benchmark(
            FAST, expected=1, environment={"SGEMM_MOCK_FAIL": "verify"},
            setup=lambda cwd: (cwd / "matrixValidationFailure.txt").symlink_to("/dev/full"))
        self.assertIn("Unable to write validation log", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="compiled CPU-only benchmark mock executable")
    arguments = parser.parse_args()
    EXECUTABLE = arguments.executable.resolve()
    if not EXECUTABLE.is_file():
        parser.error("mock executable does not exist: {}".format(EXECUTABLE))
    unittest.main(argv=[sys.argv[0]], verbosity=2)
