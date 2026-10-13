#!/usr/bin/env python3
"""Summarize a dhat heap profile: what is live at the global peak, and where
allocation churn comes from, grouped by the innermost workspace frame."""
import argparse
import json
import re
from collections import defaultdict

OWN = re.compile(r"\b(aerie\w*|akita\w*)::")
NOISE = re.compile(r"^(alloc|core|std|dhat|rayon_core|rayon|hashbrown|<alloc|<core|<std)::|^<(alloc|core|std)")


def frame_name(frame: str) -> str:
    # "0xADDR: path::to::fn (file:line:col)" -> "path::to::fn (file:line)"
    text = frame.split(": ", 1)[-1]
    text = re.sub(r"::h[0-9a-f]{16}", "", text)
    text = re.sub(r"(\S+\.rs:\d+):\d+\)", r"\1)", text)
    text = re.sub(r"\(/[^)]*/(crates/[^)]*)\)", r"(\1)", text)
    text = re.sub(r"\(/[^)]*/(akita[^)]*)\)", r"(\1)", text)
    return text


def own_frames(pp, ftbl):
    names = [frame_name(ftbl[i]) for i in pp["fs"]]
    return [n for n in names if OWN.search(n.split(" (")[0])]


def mib(value: float) -> str:
    return f"{value / 2**20:,.1f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    try:
        with open(args.profile) as profile:
            data = json.load(profile)
        pps, ftbl = data["pps"], data["ftbl"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(f"{args.profile}: not a dhat profile ({error})") from None
    peak = sum(pp["gb"] for pp in pps)
    total = sum(pp["tb"] for pp in pps)
    blocks = sum(pp["tbk"] for pp in pps)
    retained = sum(pp.get("eb", 0) for pp in pps)
    print(f"# {args.profile}")
    print(f"peak live heap {mib(peak)}; live at end {mib(retained)}; allocated {mib(total)} in {blocks:,} blocks; {len(pps):,} sites\n")

    groups = defaultdict(lambda: {"gb": 0, "eb": 0, "tb": 0, "tbk": 0, "stacks": defaultdict(int)})
    for pp in pps:
        own = own_frames(pp, ftbl)
        key = own[0] if own else "(no workspace frame)"
        group = groups[key]
        group["gb"] += pp["gb"]
        group["eb"] += pp.get("eb", 0)
        group["tb"] += pp["tb"]
        group["tbk"] += pp["tbk"]
        group["stacks"][" <- ".join(own[1 : args.depth])] += pp["gb"]

    def show(title, field, formatter):
        print(f"## {title}")
        ranked = sorted(groups.items(), key=lambda item: -item[1][field])[: args.top]
        for key, group in ranked:
            if group[field] == 0:
                break
            print(f"- {formatter(group)}  {key}")
            if field == "gb":
                for stack, size in sorted(group["stacks"].items(), key=lambda s: -s[1])[:2]:
                    if size and stack:
                        print(f"    {mib(size)} via {stack}")
        print()

    show("Live at the global peak", "gb", lambda g: f"{mib(g['gb']):>12} ({100 * g['gb'] / max(peak, 1):4.1f}%)")
    if retained:
        show("Live when the profile ended", "eb", lambda g: f"{mib(g['eb']):>12} ({100 * g['eb'] / retained:4.1f}%)")
    show("Bytes allocated (churn)", "tb", lambda g: f"{mib(g['tb']):>12} in {g['tbk']:>9,} blocks")
    show("Allocation count", "tbk", lambda g: f"{g['tbk']:>9,} blocks, {mib(g['tb'])}")


if __name__ == "__main__":
    main()
