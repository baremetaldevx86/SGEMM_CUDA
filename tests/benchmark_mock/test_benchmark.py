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
          "seed", "verified", "alpha", "beta", "m", "n", "k", "kernel_name",
          "requested_kernel", "selection_reason"]
NAMES = ["cublas-fp32", "naive", "coalesced", "shared", "block-1d", "block-2d",
         "vectorized", "bank-linearized", "bank-padded", "autotuned", "warp-tiled",
         "double-buffered", "async-double-buffered"]
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
                      alpha=0.5, beta=3.0, seed=1234, supported=None,
                      shape=None, requested=None):
        supported = SIZES if supported is None else supported
        shapes = [shape] if shape else [(size, size, size) for size in supported]
        kernels = kernel if isinstance(kernel, list) else [kernel] * len(shapes)
        requested = str(kernel) if requested is None else requested
        rows = self.read_csv(cwd / "results.csv")
        self.assertEqual(len(rows), len(shapes))
        average_seconds = (iterations + 1) / 2000.0
        for row, (m, n, k), actual in zip(rows, shapes, kernels):
            self.assertEqual(int(row["kernel"]), actual)
            self.assertEqual(row["size"], str(m) if m == n == k else "")
            self.assertEqual([int(row[key]) for key in ["m", "n", "k"]], [m, n, k])
            self.assertEqual(row["kernel_name"], NAMES[actual])
            self.assertEqual(row["requested_kernel"], requested)
            self.assertTrue(row["selection_reason"])
            self.assertIn(row["selection_reason"], result.stdout)
            self.assertIn("Requested kernel {}; selected kernel {} ({}) for M={}, N={}, K={}"
                          .format(requested, actual, NAMES[actual], m, n, k), result.stdout)
            self.assertEqual(int(row["warmup"]), warmup)
            self.assertEqual(int(row["iters"]), iterations)
            self.assertEqual(int(row["seed"]), seed)
            self.assertEqual(float(row["alpha"]), alpha)
            self.assertEqual(float(row["beta"]), beta)
            self.assertAlmostEqual(float(row["average_seconds"]), average_seconds)
            gflops = 2 * m * n * k * 1e-9 / average_seconds
            self.assertAlmostEqual(float(row["gflops"]), gflops, delta=0.000501)
            self.assertEqual(row["verified"], "true" if actual else "false")
            if m == n == k:
                self.assertIn("size: ({}).".format(m), result.stdout)
            else:
                self.assertIn("dimensions: (m={}, n={}, k={}).".format(m, n, k),
                              result.stdout)
                self.assertNotIn("size: ({}).".format(m), result.stdout)
        # String columns are always quoted, including reasons containing commas.
        for raw, row in zip((cwd / "results.csv").read_text().splitlines()[1:], rows):
            quoted = ['"{}"'.format(row[key].replace('"', '""')) for key in
                      ["kernel_name", "requested_kernel", "selection_reason"]]
            self.assertTrue(raw.endswith(",".join(quoted)))

        stats = re.findall(r"MOCK_SHAPE: m=(\d+) n=(\d+) k=(\d+) kernel=(\d+) "
                           r"launches=(\d+) resets=(\d+) timed=(\d+) verifications=(\d+)",
                           result.stdout)
        self.assertEqual([tuple(map(int, stat[:3])) for stat in stats], shapes)
        total = 0
        for stat, actual in zip(stats, kernels):
            expected_launches = warmup + iterations + (2 if actual else 0)
            self.assertEqual(list(map(int, stat[3:])),
                             [actual, expected_launches, expected_launches,
                              iterations, int(actual != 0)])
            total += expected_launches
        self.assertIn("MOCK_CHECK: launches={} resets={} timed={} transitions={}"
                      .format(total, total, iterations * len(shapes), len(shapes)),
                      result.stdout)
        am, an, ak = shape or (4096, 4096, 4096)
        self.assertIn("MOCK_MEMORY: a={} b={} c={} allocations=5 uploads=3"
                      .format(am * ak * 4, ak * an * 4, am * an * 4), result.stdout)
        self.assertIn("MOCK_DEVICE: count_queries=1 property_queries=1", result.stdout)
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
        self.assertIn("Skipping kernel 11 (11) at size 128:", result.stdout)
        self.assertIn("Kernel 11 requires", result.stdout)
        self.assertEqual(result.stdout.count("Skipping kernel"), 1)
        self.check_success(result, cwd, kernel=11, warmup=0, iterations=2,
                           supported=SIZES[1:])

    def test_list_kernels_is_standalone_and_does_not_query_cuda(self):
        result, cwd = self.run_benchmark(
            ["--list-kernels"], environment={"SGEMM_MOCK_FORBID_CUDA": "1"})
        self.assertIn("Available kernels", result.stdout)
        for kernel_id, name in enumerate(NAMES):
            self.assertIn("{}  {}".format(kernel_id, name), result.stdout)
        self.assertIn("auto - deterministic heuristic policy", result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(list(cwd.iterdir()), [])

    def test_named_selection_and_rectangular_buffers(self):
        result, cwd = self.run_benchmark(
            ["--kernel", "naive", "--m", "128", "--n", "256", "--k", "64",
             "--warmup", "0", "--iters", "1", "--csv", "results.csv"])
        self.check_success(result, cwd, kernel=1, warmup=0, iterations=1,
                           shape=(128, 256, 64), requested="naive")

    def test_auto_custom_selection_and_cublas_fallback(self):
        cases = [
            ((128, 128, 16), 10),  # first eligible policy candidate
            ((32, 32, 32), 0),    # 10, 6, and 5 reject; cuBLAS fallback
        ]
        for shape, kernel in cases:
            with self.subTest(shape=shape):
                result, cwd = self.run_benchmark(
                    ["auto", "--m", str(shape[0]), "--n", str(shape[1]),
                     "--k", str(shape[2]), "--warmup", "0", "--iters", "1",
                     "--csv", "results.csv"])
                self.check_success(result, cwd, kernel=kernel, warmup=0, iterations=1,
                                   shape=shape, requested="auto")
                self.assertIn("Auto heuristic policy", result.stdout)
                if kernel == 0:
                    self.assertIn("cuBLAS fallback", result.stdout)

    def test_explicit_unsupported_custom_shape_fails_before_allocation(self):
        result, cwd = self.run_benchmark(
            ["shared", "--m", "128", "--n", "128", "--k", "16",
             "--warmup", "0", "--iters", "1", "--csv", "results.csv"], expected=1,
            environment={"SGEMM_MOCK_FORBID_RESOURCES": "1"})
        self.assertIn("Explicit kernel 3 (shared) is unsupported", result.stderr)
        self.assertIn("multiples of 32", result.stderr)
        self.assertNotIn("MOCK_CHECK:", result.stdout)
        self.assertFalse((cwd / "results.csv").exists())

    def test_shape_options_must_be_complete_positive_and_unique(self):
        cases = [
            (["naive", "--m", "128"], "must all be specified together"),
            (["naive", "--m", "0", "--n", "1", "--k", "1"], "must be positive"),
            (["naive", "--m", "50000", "--n", "50000", "--k", "1"],
             "32-bit indexing range"),
            (["naive", "--m", "128", "--m", "128", "--n", "1", "--k", "1"],
             "Specify --m only once"),
            (["naive", "--m", "128", "--n", "1", "--k", "1", "--k", "1"],
             "Specify --k only once"),
        ]
        for args, diagnostic in cases:
            with self.subTest(args=args):
                result, _ = self.run_benchmark(
                    args, expected=1, environment={"SGEMM_MOCK_FORBID_CUDA": "1"})
                self.assertIn(diagnostic, result.stderr)
                self.assertIn("Usage:", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_all_unsupported_sizes_fail_before_allocations(self):
        result, cwd = self.run_benchmark(
            [*FAST, "--csv", "results.csv"], expected=1,
            environment={"SGEMM_MOCK_THREADS": "512", "SGEMM_MOCK_FORBID_RESOURCES": "1"})
        self.assertEqual(result.stdout.count("Skipping kernel"), len(SIZES))
        self.assertIn("No runnable benchmark cases", result.stderr)
        self.assertFalse((cwd / "results.csv").exists())
        self.assertNotIn("MOCK_CHECK:", result.stdout)

    def test_csv_is_optional(self):
        result, cwd = self.run_benchmark(FAST)
        self.assertIn("MOCK_CHECK: launches=18 resets=18 timed=6", result.stdout)
        self.assertEqual(list(cwd.iterdir()), [])

    def test_help_requires_no_device(self):
        for flag in ["--help", "-h"]:
            with self.subTest(flag=flag):
                result, _ = self.run_benchmark(
                    [flag], environment={"DEVICE": "invalid", "SGEMM_MOCK_FORBID_CUDA": "1"})
                self.assertIn("Usage:", result.stderr)
                self.assertIn("--alpha", result.stderr)
                self.assertIn("--beta", result.stderr)
                self.assertIn("reset copies are not timed", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_invalid_cli_has_useful_diagnostic(self):
        cases = [
            ([], "Please select a kernel"), (["13"], "Unknown kernel"),
            (["--kernel", "-1"], "Unknown kernel"),
            (["1", "--kernel", "2"], "only once"),
            (["--kernel", "1", "2"], "Unknown argument"),
            (["--kernel", "1", "--kernel", "1"], "only once"),
            (["bogus"], "Unknown kernel"),
            (["Naive"], "Unknown kernel"),
            (["1x"], "Unknown kernel"),
            (["99999999999999999999999"], "Unknown kernel"),
            (["--kernel", "1 "], "Unknown kernel"),
            (["1", "--warmup", "-1"], "--warmup must be non-negative"),
            (["1", "--iters", "0"], "--iters must be greater than zero"),
            (["1", "--iters", "-1"], "--iters must be greater than zero"),
            (["1", "--csv", ""], "--csv requires a non-empty path"),
            (["1", "--bogus"], "Unknown argument"),
        ]
        for option in ["--kernel", "--warmup", "--iters", "--seed", "--alpha", "--beta", "--csv",
                       "--m", "--n", "--k"]:
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
                result, _ = self.run_benchmark(
                    args, expected=1, environment={"SGEMM_MOCK_FORBID_CUDA": "1"})
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
            "device_count", "set_device", "device_properties", "event_create", "malloc", "copy", "reset",
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
