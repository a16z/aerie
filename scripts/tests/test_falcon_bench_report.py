from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
LEGACY_HEADER = HEADER.replace("records\t", "records\tprofile\t", 1)


def row(prove_ms: float = 2000.0, records: int = 16384) -> str:
    return (
        f"{records}\t100.0\t{prove_ms}\t120.0\t500.0\t280.0\t50.0\t80.0\t"
        "456\t789\t1000\ts1:1.0"
    )


def legacy_row(profile: str) -> str:
    return row().replace("16384\t", f"16384\t{profile}\t", 1)


class ParseTests(unittest.TestCase):
    def test_validates_benchmark_schema_and_build_revision(self) -> None:
        revision = "a" * 40
        output = (
            f"# benchmark_schema\t{REPORT.BENCHMARK_OUTPUT_SCHEMA}\n"
            f"# build_revision\t{revision}\n"
        )
        REPORT.validate_build_identity(output, revision)

    def test_accepts_every_supported_benchmark_schema(self) -> None:
        revision = "a" * 40
        for schema in REPORT.BENCHMARK_OUTPUT_SCHEMAS:
            output = f"# benchmark_schema\t{schema}\n# build_revision\t{revision}\n"
            REPORT.validate_build_identity(output, revision)

    def test_rejects_an_unknown_benchmark_schema(self) -> None:
        output = f"# benchmark_schema\t1\n# build_revision\t{'a' * 40}\n"
        with self.assertRaisesRegex(REPORT.BenchError, "schema"):
            REPORT.validate_build_identity(output, "a" * 40)

    def test_rejects_a_stale_benchmark_binary(self) -> None:
        output = (
            f"# benchmark_schema\t{REPORT.BENCHMARK_OUTPUT_SCHEMA}\n"
            f"# build_revision\t{'a' * 40}\n"
        )
        with self.assertRaisesRegex(REPORT.BenchError, "revision"):
            REPORT.validate_build_identity(output, "b" * 40)

    def test_rejects_an_unversioned_or_old_schema_binary(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "schema"):
            REPORT.validate_build_identity("# build_revision\tunversioned\n", "a" * 40)

    def test_validates_the_reported_parameter_set(self) -> None:
        revision = "a" * 40
        identity = (
            f"# benchmark_schema\t{REPORT.BENCHMARK_OUTPUT_SCHEMA}\n"
            f"# build_revision\t{revision}\n"
        )
        REPORT.validate_build_identity(identity, revision, "512")
        REPORT.validate_build_identity(identity + "# parameter_set\tFalcon-512\n", revision, "512")
        REPORT.validate_build_identity(
            identity + "# parameter_set\tFalcon-1024\n", revision, "1024"
        )
        with self.assertRaisesRegex(REPORT.BenchError, "did not report its parameter set"):
            REPORT.validate_build_identity(identity, revision, "1024")
        with self.assertRaisesRegex(REPORT.BenchError, "measured 'Falcon-512'"):
            REPORT.validate_build_identity(
                identity + "# parameter_set\tFalcon-512\n", revision, "1024"
            )
        with self.assertRaisesRegex(REPORT.BenchError, "measured 'Falcon-1024'"):
            REPORT.validate_build_identity(
                identity + "# parameter_set\tFalcon-1024\n", revision, "512"
            )

    def test_benchmark_command_selects_the_parameter_set(self) -> None:
        binary, time_path = Path("/bin/falcon"), Path("/tmp/time.txt")
        default = REPORT.benchmark_command(binary, time_path, 16384, "512")
        self.assertEqual(default[-2:], ["16384", REPORT.LEGACY_ISOLATION_FLAG])
        wide = REPORT.benchmark_command(binary, time_path, 16384, "1024")
        self.assertEqual(wide[-3:], ["16384", REPORT.LEGACY_ISOLATION_FLAG, "--degree=1024"])

    def test_parses_the_production_row(self) -> None:
        metrics, isolated = REPORT.parse_tsv(f"{HEADER}\n{row()}\n")
        self.assertTrue(isolated)
        self.assertNotIn("profile", metrics)
        self.assertEqual(metrics["envelope_bytes"], 1000.0)
        self.assertEqual(metrics["throughput_signatures_per_second"], 8192.0)

    def test_parses_an_isolated_legacy_centered_signed_row(self) -> None:
        metrics, isolated = REPORT.parse_tsv(
            f"{LEGACY_HEADER}\n{legacy_row('centered-signed')}\n"
        )
        self.assertTrue(isolated)
        self.assertNotIn("profile", metrics)
        self.assertEqual(metrics["envelope_bytes"], 1000.0)

    def test_legacy_extra_profile_is_not_isolated(self) -> None:
        legacy_header = LEGACY_HEADER.replace("\tenvelope_bytes", "")
        legacy_unsigned = legacy_row("unsigned-bit").replace("\t1000\ts1:1.0", "\ts1:1.0")
        legacy_centered = legacy_row("centered-signed").replace("\t1000\ts1:1.0", "\ts1:1.0")
        output = f"{legacy_header}\n{legacy_unsigned}\n{legacy_centered}\n"
        metrics, isolated = REPORT.parse_tsv(output)
        self.assertFalse(isolated)
        self.assertNotIn("envelope_bytes", metrics)

    def test_rejects_missing_target_row(self) -> None:
        with self.assertRaisesRegex(REPORT.BenchError, "expected one"):
            REPORT.parse_tsv(f"{LEGACY_HEADER}\n{legacy_row('unsigned-bit')}\n")
        with self.assertRaisesRegex(REPORT.BenchError, "expected one"):
            REPORT.parse_tsv(f"{HEADER}\n{row(records=8)}\n")

    def test_parses_gnu_time_rss(self) -> None:
        output = "Command being timed: x\nMaximum resident set size (kbytes): 123456\n"
        self.assertEqual(REPORT.parse_max_rss_kib(output), 123456)


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
            "schema_version": 1,
            "status": "success",
            "workload": {
                "records": 16384,
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
                "profile_isolated": True,
                "median": head_metrics,
            },
            "base": {
                "sha": "2" * 40,
                "profile_isolated": True,
                "median": base_metrics,
            },
            "failures": {},
        }

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

    def test_a_summary_without_a_parameter_set_renders_as_falcon_512(self) -> None:
        rendered = REPORT.render_summary(self.summary())
        self.assertIn("### Falcon benchmark — 16K honest signatures", rendered)
        self.assertIn("16,384 Falcon-512 records", rendered)

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
            "profile_isolated": False,
            "median": {},
        }
        summary["failures"] = {"base": "unsupported schedule"}
        rendered = REPORT.render_summary(summary)
        self.assertIn("merge-base benchmark failed", rendered)
        self.assertIn("| &emsp;PCS prove | n/a | 500.0 ms | n/a |", rendered)

    def test_head_failure_fails_ci(self) -> None:
        self.assertEqual(REPORT.benchmark_outcome({"head": "failed"}), ("failed", 1))

    def test_omits_legacy_process_rss(self) -> None:
        summary = self.summary()
        summary["base"]["profile_isolated"] = False  # type: ignore[index]
        rendered = REPORT.render_summary(summary)
        self.assertIn("| Peak RSS | n/a | 2.00 GiB | n/a |", rendered)
        self.assertIn("legacy harness also ran the retired", rendered)

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
