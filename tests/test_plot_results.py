#!/usr/bin/env python3
"""Square CSV parser regression tests; never render plots or rewrite artifacts."""
import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

DEPENDENCIES = ("pandas", "matplotlib", "seaborn")
missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
if missing:
    print("SKIP: plot parser tests require " + ", ".join(missing))
    sys.exit(77)

# Keep imports independent of a display server; no plotting functions are called.
os.environ["MPLBACKEND"] = "Agg"
SCRIPT = Path(__file__).resolve().parents[1] / "plot_benchmark_results.py"
spec = importlib.util.spec_from_file_location("plot_benchmark_results", SCRIPT)
plot_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_results)

LEGACY_FIELDS = ["kernel", "size", "average_seconds", "gflops", "warmup", "iters",
                 "seed", "verified", "alpha", "beta"]
FIELDS = LEGACY_FIELDS + ["m", "n", "k", "kernel_name", "requested_kernel",
                          "selection_reason"]


class PlotParserTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="sgemm-plot-parser-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "results.csv"

    def write_csv(self, fields=FIELDS, rows=None):
        if rows is None:
            rows = [self.row()]
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return self.path

    @staticmethod
    def row(**changes):
        row = dict(kernel=10, size=128, average_seconds=0.00001, gflops=419.4,
                   warmup=0, iters=1, seed=1234, verified=True, alpha=0.5,
                   beta=3.0, m=128, n=128, k=128, kernel_name="warp-tiled",
                   requested_kernel="auto", selection_reason='policy: 10, "eligible"')
        row.update(changes)
        return row

    def test_legacy_csv_schemas_remain_supported(self):
        for fields in (["kernel", "size", "gflops"], LEGACY_FIELDS):
            with self.subTest(fields=fields):
                frame = plot_results.parse_csv(self.write_csv(fields))
                self.assertEqual(list(frame.columns), ["kernel", "size", "gflops"])
                self.assertEqual(frame.to_dict("records"),
                                 [dict(kernel=10, size=128, gflops=419.4)])

    def test_extended_square_csv_uses_actual_kernel_id(self):
        rows = [self.row(), self.row(kernel=0, kernel_name="cublas-fp32",
                                    size=127, m=127, n=127, k=127)]
        frame = plot_results.parse_csv(self.write_csv(rows=rows))
        self.assertEqual(list(frame.columns), ["kernel", "size", "gflops"])
        self.assertEqual(frame["kernel"].tolist(), [10, 0])
        self.assertEqual(frame["size"].tolist(), [128, 127])

    def test_rectangular_rows_rejected_even_with_misleading_size(self):
        for dimensions in (dict(m=128, n=256, k=128),
                           dict(m=128, n=128, k=64)):
            for size in ("", 128):
                with self.subTest(dimensions=dimensions, size=size):
                    # A mixed file must fail rather than plotting only its square row.
                    rows = [self.row(), self.row(size=size, **dimensions)]
                    with self.assertRaisesRegex(
                            ValueError, r"CSV row 3: rectangular/non-cube.*M=N=K"):
                        plot_results.parse_csv(self.write_csv(rows=rows))

    def test_partial_dimension_schema_rejected(self):
        for omitted in ("m", "n", "k"):
            with self.subTest(omitted=omitted):
                fields = [field for field in FIELDS if field != omitted]
                with self.assertRaisesRegex(ValueError, "m, n, k must all be present"):
                    plot_results.parse_csv(self.write_csv(fields))

    def test_invalid_dimensions_rejected(self):
        for field in ("m", "n", "k"):
            for value in ("", "bad", 0, -1, 128.5, "inf"):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field + " must be a positive integer"):
                        plot_results.parse_csv(self.write_csv(
                            rows=[self.row(**{field: value})]))

    def test_square_size_must_match_dimensions(self):
        with self.assertRaisesRegex(ValueError, "size must match m=n=k"):
            plot_results.parse_csv(self.write_csv(rows=[self.row(size=256)]))

    def test_invalid_or_blank_sizes_rejected_with_or_without_dimensions(self):
        for fields in (FIELDS, LEGACY_FIELDS):
            for size in ("", "bad", 0, -1, 128.5, "inf"):
                with self.subTest(fields=fields, size=size):
                    with self.assertRaisesRegex(ValueError, "size must be a positive integer"):
                        plot_results.parse_csv(self.write_csv(
                            fields, rows=[self.row(size=size)]))

    def test_missing_required_columns_have_useful_diagnostic(self):
        for omitted in ("kernel", "size", "gflops"):
            with self.subTest(omitted=omitted):
                fields = [field for field in FIELDS if field != omitted]
                with self.assertRaisesRegex(ValueError, "missing required CSV columns: " + omitted):
                    plot_results.parse_csv(self.write_csv(fields))

    def test_legacy_square_text_still_parses(self):
        self.path.write_text(
            "Selected kernel: 10 (warp-tiled)\n"
            "Average elapsed time: (0.005661) s, performance: (24277.4) GFLOPS. size: (4096).\n"
        )
        self.assertEqual(plot_results.parse_file(self.path),
                         dict(size=[4096], gflops=[24277.4]))


if __name__ == "__main__":
    unittest.main()
