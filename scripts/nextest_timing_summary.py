#!/usr/bin/env python3
"""Render a compact Markdown summary from cargo-nextest JUnit output."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def test_cases(junit: Path) -> list[tuple[float, str]]:
    root = ET.parse(junit).getroot()
    cases = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "testcase":
            continue
        name = element.attrib.get("name", "<unnamed>")
        class_name = element.attrib.get("classname", "")
        qualified = f"{class_name}::{name}" if class_name else name
        try:
            duration = float(element.attrib.get("time", "0"))
        except ValueError:
            duration = 0.0
        cases.append((duration, qualified))
    return cases


def wall_seconds(timing: Path | None) -> int | None:
    if timing is None or not timing.is_file():
        return None
    data = json.loads(timing.read_text(encoding="utf-8"))
    return int(data["finished_at_epoch"]) - int(data["started_at_epoch"])


def markdown(cases: list[tuple[float, str]], wall: int | None, limit: int) -> str:
    lines = ["## Test timing", ""]
    summary = f"{len(cases)} tests; {sum(duration for duration, _ in cases):.2f}s summed test time"
    if wall is not None:
        summary += f"; {wall}s job-step wall time"
    lines.extend([summary + ".", "", "| Slowest tests | Time |", "| --- | ---: |"])
    for duration, name in sorted(cases, reverse=True)[:limit]:
        safe_name = name.replace("|", "\\|")
        lines.append(f"| `{safe_name}` | {duration:.3f}s |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit", type=Path)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(markdown(test_cases(args.junit), wall_seconds(args.timing), args.limit), end="")


if __name__ == "__main__":
    main()
