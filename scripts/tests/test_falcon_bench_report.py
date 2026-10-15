from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "falcon_bench_report.py"
SPEC = importlib.util.spec_from_file_location("falcon_bench_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


HEADER = (
    "records\tsetup_ms\tprove_ms\tcommit_ms\tpcs_prove_ms\t"
    "stage3_ms\tverify_parallel_ms\t"
    "verify_single_thread_ms\tcommitment_bytes\t"
    "pcs_proof_bytes\tenvelope_bytes\tcommit_groups_ms"
)
METADATA = {
    "benchmark_schema": REPORT.BENCHMARK_OUTPUT_SCHEMA,
    "git_revision": "a" * 40,
    "parameter_set": "Falcon-512",
    "rayon_threads": "4",
    "simd_backend": "avx2",
    "keccak_backend": "avx2",
    "rustc": "rustc 1.95.0",
    "akita_rev": "b" * 40,
    "salt_mode": "per-record",
}


def output(*, metadata: dict[str, str] | None = None, records: int = 16384) -> str:
    configuration = METADATA if metadata is None else metadata
    return "".join(f"# {key}\t{value}\n" for key, value in configuration.items()) + f"{HEADER}\n{row(records=records)}\n"



def row(prove_ms: float = 2000.0, records: int = 16384) -> str:
    return (
        f"{records}\t100.0\t{prove_ms}\t120.0\t500.0\t280.0\t50.0\t80.0\t"
        "456\t789\t1000\ts1:1.0"
    )


def output_with_cell(column: str, value: str) -> str:
    lines = output().splitlines()
    cells = lines[-1].split("\t")
    cells[HEADER.split("\t").index(column)] = value
    lines[-1] = "\t".join(cells)
    return "\n".join(lines) + "\n"


class ParseTests(unittest.TestCase):
    def test_rust_and_script_benchmark_schemas_agree(self) -> None:
        source = (SCRIPT.parents[1] / "crates/aerie-bench/benches/falcon.rs").read_text()
        match = re.search(r"const BENCHMARK_SCHEMA: u32 = ([0-9]+);", source)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), REPORT.BENCHMARK_OUTPUT_SCHEMA)

    def test_print_schema_cli(self) -> None:
        result = subprocess.run([sys.executable, str(SCRIPT), "--print-schema"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), REPORT.BENCHMARK_OUTPUT_SCHEMA)

    def test_rayon_threads_must_match_the_requested_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.CompletedProcess([], 0, output(), "")
            with patch.object(REPORT.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(REPORT.BenchError, "rayon_threads=4; expected 8"):
                    REPORT.run_process(
                        binary=Path("/bin/falcon"), side="head", phase="measured", iteration=0,
                        output_dir=Path(directory), fixture_dir=Path(directory), threads=8,
                        records=16384, expected_revision="a" * 40,
                    )

    def test_head_only_run_uses_the_degraded_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "falcon"
            binary.touch()
            args = REPORT.parser().parse_args([
                "run", "--head-binary", str(binary),
                "--output-dir", str(root / "report"), "--fixture-dir", str(root / "fixtures"),
                "--head-sha", "a" * 40, "--base-sha", "b" * 40,
                "--threads", "4", "--records", "16384", "--warmups", "1", "--runs", "2",
            ])
            sample = REPORT.parse_tsv(output())
            sample["max_rss_kib"] = 123456
            with patch.object(REPORT, "run_process", return_value=sample) as run:
                self.assertEqual(REPORT.run_benchmarks(args), 0)
                self.assertEqual(run.call_count, 3)
                self.assertTrue(all(call.kwargs["side"] == "head" for call in run.call_args_list))
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(summary["status"], "degraded")
            self.assertEqual(len(summary["head"]["samples"]), 2)
            self.assertEqual(summary["base"]["samples"], [])
            self.assertIn("no base comparison", REPORT.render_summary(summary))

    def test_validates_benchmark_schema_and_git_revision(self) -> None:
        REPORT.validate_build_identity(output(), "a" * 40)

    def test_rejects_an_unsupported_benchmark_schema(self) -> None:
        for schema in ("", "1", str(int(REPORT.BENCHMARK_OUTPUT_SCHEMA) - 1),
                       str(int(REPORT.BENCHMARK_OUTPUT_SCHEMA) + 1)):
            with self.subTest(schema=schema):
                metadata = {**METADATA, "benchmark_schema": schema}
                with self.assertRaisesRegex(REPORT.BenchError, "schema"):
                    REPORT.validate_build_identity(output(metadata=metadata), "a" * 40)
                with self.assertRaisesRegex(REPORT.BenchError, "schema"):
                    REPORT.parse_tsv(output(metadata=metadata))

    def test_rejects_a_stale_benchmark_binary(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "revision"):
            REPORT.validate_build_identity(output(), "b" * 40)

    def test_dirty_revision_identity_is_exact(self) -> None:
        revision = "a" * 40
        metadata = {**METADATA, "git_revision": revision + "-dirty"}
        with self.assertRaisesRegex(REPORT.BenchError, "revision"):
            REPORT.validate_build_identity(output(metadata=metadata), revision)
        REPORT.validate_build_identity(output(metadata=metadata), revision + "-dirty")

    def test_requires_complete_metadata(self) -> None:
        for key in REPORT.REQUIRED_METADATA:
            for value in (None, ""):
                with self.subTest(key=key, value=value):
                    metadata = dict(METADATA)
                    if value is None:
                        metadata.pop(key)
                    else:
                        metadata[key] = value
                    for parse in (REPORT.parse_tsv, lambda stdout: REPORT.validate_build_identity(stdout, "a" * 40)):
                        with self.assertRaisesRegex(REPORT.BenchError, f"missing required metadata: {key}"):
                            parse(output(metadata=metadata))

    def test_backend_metadata_is_optional(self) -> None:
        metadata = dict(METADATA)
        metadata.pop("simd_backend")
        metadata.pop("keccak_backend")
        metrics = REPORT.parse_tsv(output(metadata=metadata))
        self.assertEqual(metrics["metadata"], metadata)

    def test_rayon_thread_count_must_be_positive(self) -> None:
        for count in ("0", "-1", "four"):
            with self.subTest(count=count):
                metadata = {**METADATA, "rayon_threads": count}
                with self.assertRaisesRegex(REPORT.BenchError, "rayon_threads"):
                    REPORT.parse_tsv(output(metadata=metadata))

    def test_rejects_an_unversioned_binary(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "schema"):
            REPORT.validate_build_identity("# git_revision\tunversioned\n", "a" * 40)

    def test_validates_the_reported_parameter_set(self) -> None:
        revision = "a" * 40
        metadata = {**METADATA, "parameter_set": "Falcon-1024"}
        REPORT.validate_build_identity(output(metadata=metadata), revision, "1024")
        with self.assertRaisesRegex(REPORT.BenchError, "measured 'Falcon-512'"):
            REPORT.validate_build_identity(output(), revision, "1024")
        with self.assertRaisesRegex(REPORT.BenchError, "measured 'Falcon-1024'"):
            REPORT.validate_build_identity(output(metadata=metadata), revision, "512")
        metadata.pop("parameter_set")
        with self.assertRaisesRegex(REPORT.BenchError, "missing required metadata: parameter_set"):
            REPORT.validate_build_identity(output(metadata=metadata), revision)

    def test_benchmark_command_selects_the_parameter_set(self) -> None:
        binary, time_path = Path("/bin/falcon"), Path("/tmp/time.txt")
        default = REPORT.benchmark_command(binary, time_path, 16384, "512")
        self.assertEqual(default[-2:], [str(binary), "16384"])
        wide = REPORT.benchmark_command(binary, time_path, 16384, "1024")
        self.assertEqual(wide[-3:], [str(binary), "16384", "--degree=1024"])

    def test_parses_the_target_row_and_configuration(self) -> None:
        metrics = REPORT.parse_tsv(output())
        self.assertEqual(metrics["metadata"], METADATA)
        self.assertEqual(metrics["envelope_bytes"], 1000.0)
        self.assertEqual(metrics["throughput_signatures_per_second"], 8192.0)
        self.assertEqual(metrics["commit_groups_ms"], "s1:1.0")

    def test_rejects_nonfinite_negative_and_nonnumeric_scalar_cells(self) -> None:
        columns = HEADER.split("\t")[1:-1] + ["optional_ms"]
        for column in columns:
            for value in ("NaN", "inf", "-inf", "-1", "invalid"):
                with self.subTest(column=column, value=value):
                    if column == "optional_ms":
                        stdout = output().replace("commit_groups_ms", "optional_ms").replace("s1:1.0", value)
                    else:
                        stdout = output_with_cell(column, value)
                    with self.assertRaisesRegex(REPORT.BenchError, column):
                        REPORT.parse_tsv(stdout)

    def test_zero_scalar_cells_are_valid_except_proving_time(self) -> None:
        for column in HEADER.split("\t")[1:-1]:
            with self.subTest(column=column):
                if column == "prove_ms":
                    with self.assertRaisesRegex(REPORT.BenchError, "prove_ms must be positive"):
                        REPORT.parse_tsv(output_with_cell(column, "0"))
                else:
                    self.assertEqual(REPORT.parse_tsv(output_with_cell(column, "0"))[column], 0.0)

    def test_rejects_overflowed_throughput(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "throughput_signatures_per_second"):
            REPORT.parse_tsv(output_with_cell("prove_ms", "1e-320"))

    def test_parses_both_salt_modes(self) -> None:
        for salt_mode in ("per-record", "common"):
            with self.subTest(salt_mode=salt_mode):
                metrics = REPORT.parse_tsv(output(metadata={**METADATA, "salt_mode": salt_mode}))
                self.assertEqual(metrics["metadata"]["salt_mode"], salt_mode)
                self.assertEqual(metrics["prove_ms"], 2000.0)

    def test_invalid_measurement_fails_the_head_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "falcon"
            binary.touch()
            args = REPORT.parser().parse_args([
                "run", "--head-binary", str(binary),
                "--output-dir", str(root / "report"), "--fixture-dir", str(root / "fixtures"),
                "--head-sha", "a" * 40, "--base-sha", "b" * 40,
                "--threads", "4", "--warmups", "0", "--runs", "1",
            ])
            args.records = 16384
            completed = subprocess.CompletedProcess([], 0, output_with_cell("prove_ms", "NaN"), "")
            with patch.object(REPORT.subprocess, "run", return_value=completed):
                self.assertEqual(REPORT.run_benchmarks(args), 1)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertIn("prove_ms", summary["failures"]["head"])
            self.assertIn("PR-head benchmark failed", REPORT.render_summary(summary))

    def test_rejects_missing_target_row(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "expected one"):
            REPORT.parse_tsv(output(records=8))

    def test_rejects_duplicate_target_rows(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "expected one"):
            REPORT.parse_tsv(output() + row() + "\n")

    def test_requires_stage3_measurement(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "missing required metrics: stage3_ms"):
            REPORT.parse_tsv(output().replace("stage3_ms", "another_ms"))

    def test_requires_a_valid_salt_mode(self) -> None:
        for salt_mode in (None, "unknown"):
            metadata = dict(METADATA)
            if salt_mode is None:
                metadata.pop("salt_mode")
            else:
                metadata["salt_mode"] = salt_mode
            with self.subTest(salt_mode=salt_mode):
                message = "missing required metadata: salt_mode" if salt_mode is None else "salt mode"
                with self.assertRaisesRegex(REPORT.BenchError, message):
                    REPORT.parse_tsv(output(metadata=metadata))

    def test_parses_gnu_time_rss(self) -> None:
        time_output = "Command being timed: x\nMaximum resident set size (kbytes): 123456\n"
        self.assertEqual(REPORT.parse_max_rss_kib(time_output), 123456)

    def test_summary_carries_configuration_for_each_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "falcon"
            binary.touch()
            args = REPORT.parser().parse_args([
                "run", "--head-binary", str(binary), "--base-binary", str(binary),
                "--output-dir", str(root / "report"), "--fixture-dir", str(root / "fixtures"),
                "--head-sha", "a" * 40, "--base-sha", "a" * 40,
                "--threads", "4", "--records", "16384", "--warmups", "0", "--runs", "1",
            ])
            sample = REPORT.parse_tsv(output())
            sample["max_rss_kib"] = 123456
            with patch.object(REPORT, "run_process", return_value=sample):
                self.assertEqual(REPORT.run_benchmarks(args), 0)
            summary = json.loads((root / "report" / "summary.json").read_text())
            for side in ("head", "base"):
                self.assertEqual(summary[side]["metadata"], METADATA)
                self.assertEqual(summary[side]["samples"][0]["metadata"], METADATA)

    def test_configuration_must_be_stable_across_trials(self) -> None:
        first = REPORT.parse_tsv(output())
        second = REPORT.parse_tsv(output(metadata={**METADATA, "rayon_threads": "8"}))
        with self.assertRaisesRegex(REPORT.BenchError, "configuration changed"):
            REPORT.validate_stable_metrics([first, second])

    def test_base_schema_mismatch_skips_comparison_before_tsv_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stdout = "# benchmark_schema\t7\ninvalid TSV"
            completed = subprocess.CompletedProcess([], 0, stdout, "")
            with patch.object(REPORT.subprocess, "run", return_value=completed), patch.object(
                REPORT, "parse_tsv"
            ) as parse_tsv:
                with self.assertRaisesRegex(REPORT.BenchError, "base comparison skipped.*schema '7'"):
                    REPORT.run_process(
                        binary=Path("/bin/falcon"), side="base", phase="measured", iteration=0,
                        output_dir=Path(directory), fixture_dir=Path(directory), threads=4,
                        records=16384, expected_revision="a" * 40,
                    )
                parse_tsv.assert_not_called()


class RenderTests(unittest.TestCase):
    def summary(self) -> dict[str, object]:
        head_metrics = {
            "setup_ms": 90.0,
            "prove_ms": 1800.0,
            "commit_ms": 120.0,
            "pcs_prove_ms": 500.0,
            "stage3_ms": 280.0,
            "throughput_signatures_per_second": 9102.22,
            "verify_parallel_ms": 45.0,
            "verify_single_thread_ms": 70.0,
            "max_rss_kib": 2 * 1024 * 1024,
            "envelope_bytes": 1000.0,
            "pcs_proof_bytes": 789.0,
        }
        base_metrics = dict(head_metrics)
        base_metrics["prove_ms"] = 2000.0
        base_metrics["stage3_ms"] = 400.0
        return {
            "schema_version": REPORT.SCHEMA_VERSION,
            "status": "success",
            "workload": {
                "records": 16384,
                "parameter_set": "Falcon-512",
                "source": "reference-generated",
                "fixture_acquisition_timed": False,
            },
            "measurement": {"warmups": 1, "runs": 3},
            "environment": {
                "threads": 4,
                "runner": "ubuntu-latest",
                "cpu": "Test CPU",
                "rust_toolchain": "1.95",
            },
            "head": {
                "sha": "1" * 40,
                "metadata": dict(METADATA),
                "median": head_metrics,
            },
            "base": {
                "sha": "2" * 40,
                "metadata": dict(METADATA),
                "median": base_metrics,
            },
            "failures": {},
        }

    def test_rejects_older_summary_schema_with_bench_error(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "unsupported summary schema"):
            REPORT.render_summary({"schema_version": 1})

    def test_rejects_missing_metadata_with_bench_error(self) -> None:
        summary = self.summary()
        del summary["head"]["metadata"]
        with self.assertRaisesRegex(REPORT.BenchError, "missing configuration metadata"):
            REPORT.render_summary(summary)

    def test_measured_sides_require_valid_metadata_before_deriving_metrics(self) -> None:
        for side in ("head", "base"):
            invalid = [(None, None, "missing configuration metadata"),
                       (None, [], "missing configuration metadata")]
            invalid += [("benchmark_schema", None, "schema")]
            invalid += [(key, None, key) for key in REPORT.REQUIRED_METADATA]
            invalid += [("salt_mode", "unknown", "salt mode"),
                        ("salt_mode", ["common"], "salt_mode"),
                        ("rayon_threads", 4, "rayon_threads")]
            for key, value, message in invalid:
                with self.subTest(side=side, key=key, value=value):
                    summary = self.summary()
                    if key is None:
                        summary[side]["metadata"] = value
                    elif value is None:
                        del summary[side]["metadata"][key]
                    else:
                        summary[side]["metadata"][key] = value
                    with self.assertRaisesRegex(REPORT.BenchError, message):
                        REPORT.render_summary(summary)
                    with self.assertRaisesRegex(REPORT.BenchError, message):
                        REPORT.metric_value(summary[side], "prove_ms")

    def test_unmeasured_sides_can_omit_metadata(self) -> None:
        for side, status in (("base", "degraded"), ("head", "failed")):
            with self.subTest(side=side):
                summary = self.summary()
                summary["status"] = status
                summary[side] = {"sha": "f" * 40}
                summary["failures"] = {side: "no measurements"}
                rendered = REPORT.render_summary(summary)
                self.assertIn("no measurements", rendered)
                self.assertIsNone(REPORT.metric_value(summary[side], "prove_ms"))

    def test_summary_threads_must_match_metadata(self) -> None:
        summary = self.summary()
        summary["head"]["metadata"]["rayon_threads"] = "8"
        with self.assertRaisesRegex(REPORT.BenchError, "rayon_threads"):
            REPORT.render_summary(summary)

    def test_renders_compact_comparison(self) -> None:
        rendered = REPORT.render_summary(self.summary())
        self.assertIn(REPORT.COMMENT_MARKER, rendered)
        self.assertIn("| **Total proving** | 2,000.0 ms | 1,800.0 ms | -10.0% |", rendered)
        self.assertIn("| &emsp;PIOP stage 3 | 400.0 ms | 280.0 ms | -30.0% |", rendered)
        self.assertIn("| &emsp;PCS prove | 500.0 ms | 500.0 ms | 0.0% |", rendered)
        self.assertNotIn("Wire proof bytes", rendered)
        self.assertIn("16,384 Falcon-512 records", rendered)

    def test_renders_the_falcon_1024_leg_under_its_own_marker(self) -> None:
        summary = self.summary()
        summary["workload"]["parameter_set"] = "Falcon-1024"  # type: ignore[index]
        rendered = REPORT.render_summary(summary)
        self.assertIn(REPORT.PARAMETER_SETS["1024"]["marker"], rendered)
        self.assertNotIn(REPORT.COMMENT_MARKER, rendered)
        self.assertIn("Falcon-1024 benchmark", rendered)
        self.assertIn("16,384 Falcon-1024 records", rendered)
        self.assertIn("honest fixture", rendered)
        self.assertIn("not production key material", rendered)

    def test_groups_witness_generation_and_splits_the_proof_size(self) -> None:
        summary = self.summary()
        for side in ("head", "base"):
            summary[side]["median"].update(
                {
                    "projection_ms": 100.0,
                    "stage1_ms": 200.0,
                    "stage2_ms": 300.0,
                    "transport_materialize_ms": 1.0,
                    "transport_binding_ms": 2.0,
                    "output_encode_ms": 3.0,
                    "envelope_bytes": 16384 * 40 + 456 + 797 + 100.0,
                    "commitment_bytes": 456.0,
                    "pcs_proof_bytes": 789.0,
                }
            )
        rendered = REPORT.render_summary(summary)
        # 2000 - (100 + 200 + 300 + 400 + 500) - 6 on the base side.
        self.assertIn(
            "| &emsp;Witness generation + PCS commit | 494.0 ms | 414.0 ms | -16.2% |",
            rendered,
        )
        self.assertIn("| &emsp;Salts | 655,360 B | 655,360 B | 0.0% |", rendered)
        self.assertIn("| &emsp;PIOP | 100 B | 100 B | 0.0% |", rendered)
        self.assertIn("| &emsp;PCS commitments | 456 B | 456 B | 0.0% |", rendered)
        self.assertIn("| &emsp;PCS opening | 797 B | 797 B | 0.0% |", rendered)

    def test_missing_parts_render_as_unavailable(self) -> None:
        rendered = REPORT.render_summary(self.summary())
        self.assertIn("| &emsp;Witness generation + PCS commit | n/a | n/a | n/a |", rendered)
        self.assertIn("| &emsp;PIOP | n/a | n/a | n/a |", rendered)

    def test_renders_verifier_memory_in_mebibytes(self) -> None:
        summary = self.summary()
        summary["head"]["median"].update(
            {
                "verifier_setup_heap_bytes": 13 * 1024 * 1024,
                "verify_peak_heap_bytes": 35.25 * 1024 * 1024,
                "verify_single_thread_peak_heap_bytes": 35.0 * 1024 * 1024,
            }
        )
        rendered = REPORT.render_summary(summary)
        self.assertIn("| Setup memory | n/a | 13.0 MiB | n/a |", rendered)
        self.assertIn("| Peak working memory, all threads | n/a | 35.2 MiB | n/a |", rendered)
        self.assertIn("| Peak working memory, one thread | n/a | 35.0 MiB | n/a |", rendered)

    def test_base_only_failure_is_degraded_and_does_not_fail_ci(self) -> None:
        self.assertEqual(REPORT.benchmark_outcome({"base": "unsupported schedule"}), ("degraded", 0))
        summary = self.summary()
        summary["status"] = "degraded"
        summary["base"] = {
            "sha": "2" * 40,
            "metadata": {},
            "median": {},
        }
        summary["failures"] = {"base": "unsupported schedule"}
        rendered = REPORT.render_summary(summary)
        self.assertIn("merge-base benchmark failed", rendered)
        self.assertIn("| &emsp;PCS prove | n/a | 500.0 ms | n/a |", rendered)

    def test_head_failure_fails_ci(self) -> None:
        self.assertEqual(REPORT.benchmark_outcome({"head": "failed"}), ("failed", 1))

    def test_common_salt_is_counted_once(self) -> None:
        summary = self.summary()
        for side in ("head", "base"):
            summary[side]["metadata"]["salt_mode"] = "common"
            summary[side]["median"].update({
                "envelope_bytes": 40 + 456 + 797 + 100.0,
                "commitment_bytes": 456.0,
                "pcs_proof_bytes": 789.0,
            })
        rendered = REPORT.render_summary(summary)
        self.assertIn("| &emsp;Salts | 40 B | 40 B | 0.0% |", rendered)
        self.assertIn("| &emsp;PIOP | 100 B | 100 B | 0.0% |", rendered)

    def test_summary_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(self.summary()), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "success")

    def test_render_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(self.summary()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "render", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Falcon benchmark", completed.stdout)


if __name__ == "__main__":
    unittest.main()
