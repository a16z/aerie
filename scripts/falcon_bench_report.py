#!/usr/bin/env python3
"""Run and render the fixed Aerie Falcon PR benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
# Schemas whose TSV carries every REQUIRED_METRICS column. Schema 3 only
# dropped the optional verify_ring_keys_ms column, so a schema-2 base binary
# still compares cleanly against a schema-3 head. Schema 4 dropped the
# `profile` column when the non-production cell encoding was removed; older
# binaries still print it, and their production row is LEGACY_PROFILE.
# Schema 6 reports the three stages and commitment groups. Schema 7 drops the
# unused statement-binding timing column.
BENCHMARK_OUTPUT_SCHEMAS = ("2", "3", "4", "5", "6", "7")
BENCHMARK_OUTPUT_SCHEMA = BENCHMARK_OUTPUT_SCHEMAS[-1]
WORKLOAD_RECORDS = 16_384
LEGACY_PROFILE = "centered-signed"
# Schema-2 and schema-3 binaries also run the retired unsigned/bit encoding
# in the same process unless given this flag. Schema-4 binaries ignore
# unrecognized flags, so the command passes it to base and head alike.
LEGACY_ISOLATION_FLAG = "--centered-signed-only"
COMMENT_MARKER = "<!-- aerie-falcon-bench-report -->"
# One CI leg per parameter set. The TSV contract is shared: the parameter
# set travels in the existing `# parameter_set` metadata line (printed by
# schema-2 and schema-3 binaries alike), so no new output schema is needed.
# The Falcon-1024 leg runs honest FN-DSA-1024 fixture signatures from the
# reference `fn-dsa` crate with deterministic benchmark randomness; it is a
# fixture benchmark, not a production key-material measurement.
PARAMETER_SETS: dict[str, dict[str, Any]] = {
    "512": {
        "label": "Falcon-512",
        "records": WORKLOAD_RECORDS,
        "binary_args": (),
        "marker": COMMENT_MARKER,
        "title": "Falcon benchmark — 16K honest signatures",
        "source": "distinct reference-generated FN-DSA keypairs and signatures",
    },
    "1024": {
        "label": "Falcon-1024",
        "records": WORKLOAD_RECORDS,
        "binary_args": ("--degree=1024",),
        "marker": "<!-- aerie-falcon-1024-bench-report -->",
        "title": "Falcon-1024 benchmark — 16K honest-fixture signatures",
        "source": (
            "distinct reference-generated FN-DSA-1024 keypairs and signatures "
            "(honest fixture; deterministic benchmark randomness, not production "
            "key material)"
        ),
    },
}
DEFAULT_PARAMETER_SET = "512"
REQUIRED_METRICS = (
    "setup_ms",
    "prove_ms",
    "commit_ms",
    "pcs_prove_ms",
    "verify_parallel_ms",
    "verify_single_thread_ms",
    "commitment_bytes",
    "pcs_proof_bytes",
)
STABLE_METRICS = (
    "commitment_bytes",
    "pcs_proof_bytes",
    "envelope_bytes",
)


class BenchError(RuntimeError):
    """A benchmark process or its output violated the measurement contract."""


def validate_build_identity(
    stdout: str, expected_revision: str, parameter_set: str = DEFAULT_PARAMETER_SET
) -> None:
    """Reject stale, wrong-revision, or wrong-parameter-set executables."""
    metadata: dict[str, str] = {}
    for line in stdout.splitlines():
        if not line.startswith("# ") or "\t" not in line:
            continue
        key, value = line[2:].split("\t", 1)
        metadata[key] = value
    if metadata.get("benchmark_schema") not in BENCHMARK_OUTPUT_SCHEMAS:
        raise BenchError(
            "benchmark executable reported schema "
            f"{metadata.get('benchmark_schema')!r}; expected one of {BENCHMARK_OUTPUT_SCHEMAS}"
        )
    if metadata.get("build_revision") != expected_revision:
        raise BenchError(
            "benchmark executable revision "
            f"{metadata.get('build_revision')!r}; expected {expected_revision}"
        )
    # A binary that ignored the parameter-set flag would silently measure the
    # wrong set, so any set but the default must be reported explicitly.
    expected_label = PARAMETER_SETS[parameter_set]["label"]
    reported = metadata.get("parameter_set")
    if reported is None and parameter_set != DEFAULT_PARAMETER_SET:
        raise BenchError(
            f"benchmark executable did not report its parameter set; expected {expected_label}"
        )
    if reported is not None and reported != expected_label:
        raise BenchError(
            f"benchmark executable measured {reported!r}; expected {expected_label}"
        )


def parse_tsv(stdout: str, records: int = WORKLOAD_RECORDS) -> tuple[dict[str, Any], bool]:
    """Parse the production row and report whether it was the only row.

    A legacy (schema 2 or 3) harness also prints a `profile` column and may run
    a second encoding in the same process; only its LEGACY_PROFILE row counts.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("records\t")),
        None,
    )
    if header_index is None:
        raise BenchError("benchmark output did not contain the TSV header")

    header = lines[header_index].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        if line.startswith("#"):
            continue
        values = line.split("\t")
        if len(values) != len(header):
            raise BenchError(
                f"benchmark row has {len(values)} columns; expected {len(header)}"
            )
        rows.append(dict(zip(header, values, strict=True)))

    legacy = "profile" in header
    matches = [
        row
        for row in rows
        if row.get("records") == str(records)
        and (not legacy or row.get("profile") == LEGACY_PROFILE)
    ]
    if len(matches) != 1:
        raise BenchError(f"expected one {records}-record production row, found {len(matches)}")

    parsed: dict[str, Any] = {"records": records}
    for key, value in matches[0].items():
        if key in {"records", "profile"}:
            continue
        if key in {"commit_groups_ms", "commit_shards_ms"}:
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value)
        except ValueError as error:
            raise BenchError(f"non-numeric value for {key}: {value!r}") from error

    # Stage timings are distinct measurements across the cutover; never
    # relabel the old final block as the new third stage.
    stage_metric = "stage3_ms" if "stage3_ms" in header or "# benchmark_schema\t6" in lines else "f_ms"
    missing = [key for key in (*REQUIRED_METRICS, stage_metric) if key not in parsed]
    if missing:
        raise BenchError(f"benchmark row is missing required metrics: {', '.join(missing)}")
    if parsed["prove_ms"] <= 0:
        raise BenchError("prove_ms must be positive")
    parsed["throughput_signatures_per_second"] = records * 1_000.0 / parsed["prove_ms"]
    return parsed, len(rows) == 1


def parse_max_rss_kib(time_output: str) -> int:
    prefix = "Maximum resident set size (kbytes):"
    for line in time_output.splitlines():
        if line.strip().startswith(prefix):
            value = line.split(":", 1)[1].strip()
            try:
                return int(value)
            except ValueError as error:
                raise BenchError(f"invalid maximum RSS value: {value!r}") from error
    raise BenchError("GNU time output did not contain maximum RSS")


def median_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {}
    numeric_keys = set.intersection(
        *(
            {
                key
                for key, value in sample.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            for sample in samples
        )
    )
    return {
        key: float(statistics.median(float(sample[key]) for sample in samples))
        for key in sorted(numeric_keys)
    }


def validate_stable_metrics(samples: list[dict[str, Any]]) -> None:
    for key in STABLE_METRICS:
        values = {sample[key] for sample in samples if key in sample}
        if len(values) > 1:
            raise BenchError(f"{key} changed across measured trials: {sorted(values)}")


def benchmark_outcome(failures: dict[str, str]) -> tuple[str, int]:
    """Classify failures without making a broken merge base block its repair."""
    if "head" in failures:
        return "failed", 1
    if "base" in failures:
        return "degraded", 0
    return "success", 0


def benchmark_command(
    binary: Path, time_path: Path, records: int, parameter_set: str
) -> list[str]:
    return [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_path),
        str(binary),
        str(records),
        LEGACY_ISOLATION_FLAG,
        *PARAMETER_SETS[parameter_set]["binary_args"],
    ]


def run_process(
    *,
    binary: Path,
    side: str,
    phase: str,
    iteration: int,
    output_dir: Path,
    fixture_dir: Path,
    threads: int,
    records: int,
    expected_revision: str,
    parameter_set: str = DEFAULT_PARAMETER_SET,
) -> dict[str, Any]:
    sample_dir = output_dir / "raw" / side
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{phase}-{iteration + 1}"
    stdout_path = sample_dir / f"{stem}.stdout.tsv"
    stderr_path = sample_dir / f"{stem}.stderr.txt"
    time_path = sample_dir / f"{stem}.time.txt"

    env = os.environ.copy()
    env["AERIE_HONEST_FIXTURE_DIR"] = str(fixture_dir)
    env["RAYON_NUM_THREADS"] = str(threads)
    env.setdefault("RUST_MIN_STACK", str(512 * 1024 * 1024))
    command = benchmark_command(binary, time_path, records, parameter_set)
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr)[-1_200:].strip()
        raise BenchError(f"{side} {phase} {iteration + 1} exited {completed.returncode}: {tail}")

    validate_build_identity(completed.stdout, expected_revision, parameter_set)
    metrics, profile_isolated = parse_tsv(completed.stdout, records)
    metrics["max_rss_kib"] = parse_max_rss_kib(time_path.read_text(encoding="utf-8"))
    metrics["profile_isolated"] = profile_isolated
    return metrics


def run_benchmarks(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    fixture_dir = Path(args.fixture_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    binaries = {
        "head": Path(args.head_binary).resolve(),
        "base": Path(args.base_binary).resolve(),
    }
    for side, binary in binaries.items():
        if not binary.is_file():
            raise BenchError(f"{side} binary does not exist: {binary}")

    measured: dict[str, list[dict[str, Any]]] = {"head": [], "base": []}
    warmup_samples: dict[str, list[dict[str, Any]]] = {"head": [], "base": []}
    failures: dict[str, str] = {}

    phases = (("warmup", args.warmups), ("measured", args.runs))
    for phase, count in phases:
        for iteration in range(count):
            order = ("base", "head") if iteration % 2 == 0 else ("head", "base")
            for side in order:
                if side in failures:
                    continue
                try:
                    sample = run_process(
                        binary=binaries[side],
                        side=side,
                        phase=phase,
                        iteration=iteration,
                        output_dir=output_dir,
                        fixture_dir=fixture_dir,
                        threads=args.threads,
                        records=args.records,
                        expected_revision=(
                            args.head_sha if side == "head" else args.base_sha
                        ),
                        parameter_set=args.parameter_set,
                    )
                except (BenchError, OSError) as error:
                    failures[side] = str(error)
                    continue
                target = warmup_samples if phase == "warmup" else measured
                target[side].append(sample)

    for side, samples in measured.items():
        if side in failures:
            continue
        if len(samples) != args.runs:
            failures[side] = f"collected {len(samples)} of {args.runs} measured trials"
            continue
        try:
            validate_stable_metrics(samples)
        except BenchError as error:
            failures[side] = str(error)

    sides: dict[str, Any] = {}
    for side in ("head", "base"):
        samples = measured[side]
        sides[side] = {
            "sha": args.head_sha if side == "head" else args.base_sha,
            "profile_isolated": bool(samples)
            and all(bool(sample["profile_isolated"]) for sample in samples),
            "warmup_samples": warmup_samples[side],
            "samples": samples,
            "median": median_metrics(samples),
        }

    status, exit_code = benchmark_outcome(failures)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "workload": {
            "records": args.records,
            "parameter_set": PARAMETER_SETS[args.parameter_set]["label"],
            "source": PARAMETER_SETS[args.parameter_set]["source"],
            "fixture_acquisition_timed": False,
        },
        "measurement": {
            "warmups": args.warmups,
            "runs": args.runs,
            "order": "interleaved, alternating first side",
        },
        "environment": {
            "threads": args.threads,
            "runner": args.runner,
            "cpu": args.cpu,
            "rust_toolchain": args.rust_toolchain,
        },
        "head": sides["head"],
        "base": sides["base"],
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return exit_code


def format_ms(value: float) -> str:
    return f"{value:,.1f} ms"


def format_count(value: float) -> str:
    return f"{value:,.0f}"


def format_bytes(value: float) -> str:
    return f"{value:,.0f} B"


def format_heap(value: float) -> str:
    return f"{value / (1024.0 * 1024.0):,.1f} MiB"


def format_rss(value: float) -> str:
    gib = value / (1024.0 * 1024.0)
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    return f"{value / 1024.0:.1f} MiB"


def delta(head: float | None, base: float | None) -> str:
    if head is None or base is None:
        return "n/a"
    if base == 0:
        return "unchanged" if head == 0 else "n/a"
    change = (head / base - 1.0) * 100.0
    if math.isclose(change, 0.0, abs_tol=0.05):
        return "0.0%"
    return f"{change:+.1f}%"


# Wire framing (crates/aerie-falcon/src/protocol/wire.rs): each per-record salt
# is 40 bytes, and the PCS opening string carries an 8-byte length prefix. The
# PIOP share is the remainder, so it includes the 27-byte header.
SALT_BYTES = 40
PCS_OPENING_LENGTH_PREFIX_BYTES = 8
PIOP_AND_OPENING_PARTS = (
    "projection_ms",
    "stage1_ms",
    "stage2_ms",
    "stage3_ms",
    "pcs_prove_ms",
)
PROVE_OTHER_PARTS = ("transport_materialize_ms", "transport_binding_ms", "output_encode_ms")
INDENT = "&emsp;"


def sum_of(median: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    if any(key not in median for key in keys):
        return None
    return float(sum(median[key] for key in keys))


def derived_metrics(median: dict[str, Any], records: int) -> dict[str, float]:
    """Report-only groupings of the benchmark's measured columns."""
    derived: dict[str, float] = {}
    other = sum_of(median, PROVE_OTHER_PARTS)
    if other is not None:
        derived["prove_other_ms"] = other
    # Witness generation and the first commitments run concurrently, so their
    # joint wall time is what total proving leaves after the later phases.
    later = sum_of(median, PIOP_AND_OPENING_PARTS)
    if other is not None and later is not None and "prove_ms" in median:
        derived["witness_and_commit_ms"] = median["prove_ms"] - later - other
    if "pcs_proof_bytes" in median:
        derived["pcs_opening_bytes"] = median["pcs_proof_bytes"] + PCS_OPENING_LENGTH_PREFIX_BYTES
    salts = records * SALT_BYTES
    derived["salt_bytes"] = float(salts)
    if all(key in median for key in ("envelope_bytes", "commitment_bytes", "pcs_proof_bytes")):
        piop = (
            median["envelope_bytes"]
            - salts
            - median["commitment_bytes"]
            - derived["pcs_opening_bytes"]
        )
        if piop > 0:
            derived["piop_bytes"] = float(piop)
    return derived


def metric_value(side: dict[str, Any], key: str, records: int = WORKLOAD_RECORDS) -> float | None:
    if key == "max_rss_kib" and not side.get("profile_isolated", False):
        return None
    median = side.get("median", {})
    if not median:
        return None
    value = median.get(key, derived_metrics(median, records).get(key))
    return float(value) if value is not None else None


def render_summary(summary: dict[str, Any]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise BenchError(f"unsupported summary schema: {summary.get('schema_version')!r}")

    head = summary["head"]
    base = summary["base"]
    set_label = summary["workload"].get("parameter_set", "Falcon-512")
    parameter_set = next(
        (key for key, value in PARAMETER_SETS.items() if value["label"] == set_label), None
    )
    if parameter_set is None:
        raise BenchError(f"unsupported summary parameter set: {set_label!r}")
    leg = PARAMETER_SETS[parameter_set]
    throughput = lambda value: f"{format_count(value)} sig/s"  # noqa: E731
    sections = (
        (
            "Prover",
            (
                ("PCS setup (once per batch size)", "setup_ms", format_ms),
                ("**Total proving**", "prove_ms", format_ms),
                (f"{INDENT}Witness generation + PCS commit", "witness_and_commit_ms", format_ms),
                (f"{INDENT}PIOP projection", "projection_ms", format_ms),
                (f"{INDENT}PIOP stage 1", "stage1_ms", format_ms),
                (f"{INDENT}PIOP stage 2", "stage2_ms", format_ms),
                (f"{INDENT}PIOP stage 3", "stage3_ms", format_ms),
                (f"{INDENT}PCS prove", "pcs_prove_ms", format_ms),
                (f"{INDENT}Other (input binding, encoding)", "prove_other_ms", format_ms),
                ("Throughput", "throughput_signatures_per_second", throughput),
                ("Peak RSS", "max_rss_kib", format_rss),
            ),
        ),
        (
            "Verifier",
            (
                ("Verification, all threads", "verify_parallel_ms", format_ms),
                ("Verification, one thread", "verify_single_thread_ms", format_ms),
                ("Setup memory", "verifier_setup_heap_bytes", format_heap),
                ("Peak working memory, all threads", "verify_peak_heap_bytes", format_heap),
                (
                    "Peak working memory, one thread",
                    "verify_single_thread_peak_heap_bytes",
                    format_heap,
                ),
            ),
        ),
        (
            "Proof size",
            (
                ("**Proof size**", "envelope_bytes", format_bytes),
                (f"{INDENT}Salts", "salt_bytes", format_bytes),
                (f"{INDENT}PIOP", "piop_bytes", format_bytes),
                (f"{INDENT}PCS commitments", "commitment_bytes", format_bytes),
                (f"{INDENT}PCS opening", "pcs_opening_bytes", format_bytes),
            ),
        ),
    )
    records = int(summary["workload"].get("records", WORKLOAD_RECORDS))

    lines = [leg["marker"], "", f"### {leg['title']}", ""]
    if summary.get("status") == "failed":
        lines.extend(
            [
                "> [!WARNING]",
                "> The PR-head benchmark failed. Available measurements are shown below.",
                "",
            ]
        )
    elif summary.get("status") == "degraded":
        lines.extend(
            [
                "> [!NOTE]",
                "> The merge-base benchmark failed, but the PR head completed successfully. "
                "Base values and deltas are unavailable; this does not block a repair for a "
                "pre-existing base failure.",
                "",
            ]
        )
    for title, rows in sections:
        lines.extend([f"| {title} | Merge base | Head | Δ |", "|---|---:|---:|---:|"])
        for label, key, formatter in rows:
            head_value = metric_value(head, key, records)
            base_value = metric_value(base, key, records)
            lines.append(
                f"| {label} | "
                f"{formatter(base_value) if base_value is not None else 'n/a'} | "
                f"{formatter(head_value) if head_value is not None else 'n/a'} | "
                f"{delta(head_value, base_value)} |"
            )
        lines.append("")
    lines.extend(
        [
            "Witness generation and the PCS commit overlap, so their row is total "
            "proving minus the rows below it. Stage 2 includes stage-3 preparation, which runs alongside it. The PIOP "
            "share of the proof includes the 27-byte header; the PCS opening includes "
            "its 8-byte length prefix. Verifier memory is heap counted by the benchmark's "
            "allocator: setup memory is what a standalone verifier setup keeps, and "
            "working memory is the most one verification holds at once beyond its "
            "inputs (public keys, messages, proof) and the setup.",
        ]
    )

    measurement = summary["measurement"]
    environment = summary["environment"]
    workload = summary["workload"]
    lines.extend(
        [
            "",
            f"Head `{head.get('sha', '')[:12]}` vs merge base `{base.get('sha', '')[:12]}`. "
            f"Medians of {measurement['runs']} measured runs after "
            f"{measurement['warmups']} warmup; "
            f"{environment['threads']} threads on {environment['cpu'] or environment['runner']}.",
            "",
            f"Workload: {workload['records']:,} {set_label} records from "
            f"{leg['source']}. Fixture acquisition is "
            "outside setup/prove/verify timing. Every measured process verifies the produced "
            "proof with both the all-threads and one-thread verifier.",
        ]
    )
    if not head.get("profile_isolated", False) or not base.get("profile_isolated", False):
        lines.extend(
            [
                "",
                "Peak RSS is `n/a` for a revision whose legacy harness also ran the retired "
                "unsigned/bit encoding in the same process; reporting that process peak as "
                "production memory would be misleading.",
            ]
        )
    failures = summary.get("failures", {})
    if failures:
        lines.extend(["", "Failed sides:"])
        for side, message in sorted(failures.items()):
            safe_message = str(message).replace("`", "'").replace("\n", " ")
            lines.append(f"- `{side}`: {safe_message}")
    return "\n".join(lines) + "\n"


def render_command(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    sys.stdout.write(render_summary(summary))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run interleaved head/base benchmarks")
    run_parser.add_argument("--head-binary", required=True)
    run_parser.add_argument("--base-binary", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--fixture-dir", required=True)
    run_parser.add_argument("--head-sha", required=True)
    run_parser.add_argument("--base-sha", required=True)
    run_parser.add_argument("--threads", required=True, type=int)
    run_parser.add_argument("--runner", default="")
    run_parser.add_argument("--cpu", default="")
    run_parser.add_argument("--rust-toolchain", default="")
    run_parser.add_argument("--records", type=int, default=None)
    run_parser.add_argument(
        "--parameter-set", choices=tuple(PARAMETER_SETS), default=DEFAULT_PARAMETER_SET
    )
    run_parser.add_argument("--warmups", type=int, default=1)
    run_parser.add_argument("--runs", type=int, default=3)
    run_parser.set_defaults(function=run_benchmarks)

    render_parser = subparsers.add_parser("render", help="render a benchmark summary as Markdown")
    render_parser.add_argument("summary")
    render_parser.set_defaults(function=render_command)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        leg = PARAMETER_SETS[args.parameter_set]
        if args.records is None:
            args.records = leg["records"]
        if args.records != leg["records"]:
            raise BenchError(
                f"{leg['label']} CI workload is fixed at {leg['records']} records"
            )
        if args.threads < 2:
            raise BenchError("Falcon CI requires at least two benchmark threads")
        if args.warmups < 0 or args.runs <= 0:
            raise BenchError("warmups must be nonnegative and runs must be positive")
    return int(args.function(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
