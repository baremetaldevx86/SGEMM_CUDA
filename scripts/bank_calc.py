#!/usr/bin/env python3
"""Compare shared-memory bank use with and without one padding column.

Run ``python3 scripts/bank_calc.py --help`` for the access-pattern options.
The model uses one 32-thread warp, 32 banks, and one 32-bit word per access.
Logical coordinates use --columns; physical row spacing uses --stride.
Each output tuple is (lane, row, column, bank). A conflict factor of 1 means
conflict-free access. No CUDA or third-party packages are needed.
"""

import argparse


WARP_SIZE = 32
BANK_COUNT = 32


def positive_int(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def printBankConflicts(row_stride, columns, items_per_thread, steps):
    for step in range(steps):
        accesses = []
        counts = [0] * BANK_COUNT
        for lane in range(WARP_SIZE):
            # Include the step when computing both coordinates: an access may
            # cross a logical row boundary when columns is not a multiple of
            # items_per_thread.
            row, col = divmod(lane * items_per_thread + step, columns)
            bank = (row * row_stride + col) % BANK_COUNT
            accesses.append((lane, row, col, bank))
            counts[bank] += 1

        print("Step", step)
        for access in accesses:
            print("(" + ",".join(str(value) for value in access) + ")")
        banks_accessed = sum(count > 0 for count in counts)
        print(
            f"Bank conflicts (Step {step}): {max(counts)}, "
            f"banks accessed: {banks_accessed}/{BANK_COUNT}\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--columns", type=positive_int, default=16,
        help="logical columns in the access pattern (default: 16)",
    )
    parser.add_argument(
        "--stride", type=positive_int, default=32,
        help="unpadded physical row stride in 32-bit words (default: 32)",
    )
    parser.add_argument(
        "--items-per-thread", type=positive_int, default=8,
        help="consecutive words assigned to each lane (default: 8)",
    )
    parser.add_argument(
        "--steps", type=positive_int, default=1,
        help="number of per-lane words to analyze (default: 1)",
    )
    args = parser.parse_args()
    if args.columns > args.stride:
        parser.error("--stride must be at least --columns")
    if args.steps > args.items_per_thread:
        parser.error("--steps must not exceed --items-per-thread")

    for label, stride in (("NAIVE", args.stride), ("EXTRA COL", args.stride + 1)):
        print(f"---{label}---")
        printBankConflicts(stride, args.columns, args.items_per_thread, args.steps)


if __name__ == "__main__":
    main()
