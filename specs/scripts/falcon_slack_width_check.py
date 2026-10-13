#!/usr/bin/env python3
"""Exhaustively check capped four-square slack decompositions.

By default this checks every delta in (8191^2, 70_265_242]. An optional
inclusive --range START END selects a subrange of that interval.
"""

import argparse
import math

CAP = 8191
LOW = CAP * CAP + 1
HIGH = 70_265_242


def is_sum_of_at_most_three_squares(value):
    """Legendre's criterion; zeros pad representations to three squares."""
    if value < 0:
        return False
    while value and value % 4 == 0:
        value //= 4
    return value % 8 != 7


def first_capped_component(delta):
    """Return a valid first component and the number of candidates examined."""
    first = min(math.isqrt(delta), CAP)
    for count, component in enumerate(range(first, -1, -1), start=1):
        remainder = delta - component * component
        if math.isqrt(remainder) > CAP:
            continue
        # Each term in a three-square representation is at most sqrt(remainder).
        if is_sum_of_at_most_three_squares(remainder):
            return component, count
    return None, first + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--range", nargs=2, type=int, metavar=("START", "END"),
        help=f"inclusive delta range within [{LOW}, {HIGH}] (default: full range)",
    )
    args = parser.parse_args()
    start, end = args.range if args.range else (LOW, HIGH)
    if start < LOW or end > HIGH or start > end:
        parser.error(f"range must satisfy {LOW} <= START <= END <= {HIGH}")

    attempts = 0
    first_choice_successes = 0
    failures = []
    for delta in range(start, end + 1):
        first = min(math.isqrt(delta), CAP)
        component, examined = first_capped_component(delta)
        attempts += examined
        if component is None:
            failures.append(delta)
            continue
        if component == first:
            first_choice_successes += 1
        remainder = delta - component * component
        if (component > CAP or math.isqrt(remainder) > CAP
                or not is_sum_of_at_most_three_squares(remainder)):
            failures.append(delta)

    count = end - start + 1
    print(f"checked deltas [{start}, {end}] ({count} values)")
    print(f"capped first square <= {CAP}; remainder has a three-square representation")
    print(f"successful first-choice decompositions: {first_choice_successes}/{count}")
    print(f"candidate first components examined: {attempts}")
    print(f"failures: {len(failures)}")
    if failures:
        print("first failing deltas: " + ", ".join(map(str, failures[:20])))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
