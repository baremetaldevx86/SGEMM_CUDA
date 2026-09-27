#!/usr/bin/env python3
"""Dependency-free regression checks for the scalar-word bank calculator."""
import re
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bank_calc.py"


class BankCalculatorTests(unittest.TestCase):
    def invoke(self, *arguments, status=0):
        result = subprocess.run([sys.executable, str(SCRIPT), *arguments],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, status, result.stderr)
        return result.stdout

    def test_default_pattern(self):
        factors = re.findall(r"Bank conflicts \(Step 0\): (\d+)", self.invoke())
        self.assertEqual(factors, ["16", "2"])

    def test_transposed_pattern(self):
        output = self.invoke("--columns", "32", "--stride", "32",
                             "--items-per-thread", "32", "--steps", "2")
        factors = re.findall(r"Bank conflicts \(Step \d+\): (\d+)", output)
        self.assertEqual(factors, ["32", "32", "1", "1"])

    def test_row_crossings(self):
        output = self.invoke("--columns", "3", "--stride", "5",
                             "--items-per-thread", "4", "--steps", "4")
        accesses = re.findall(r"\((\d+),(\d+),(\d+),(\d+)\)", output)
        self.assertEqual(len(accesses), 2 * 4 * 32)
        for index, access in enumerate(accesses):
            lane, row, col, bank = map(int, access)
            step = (index // 32) % 4
            stride = 5 + index // (4 * 32)
            self.assertEqual((row, col), divmod(lane * 4 + step, 3))
            self.assertEqual(bank, (row * stride + col) % 32)

    def test_invalid_arguments(self):
        for option in ("--columns", "--stride", "--items-per-thread", "--steps"):
            for value in ("0", "-1", "abc"):
                with self.subTest(option=option, value=value):
                    self.invoke(option, value, status=2)
        self.invoke("--columns", "33", "--stride", "32", status=2)
        self.invoke("--items-per-thread", "1", "--steps", "2", status=2)


if __name__ == "__main__":
    unittest.main()
