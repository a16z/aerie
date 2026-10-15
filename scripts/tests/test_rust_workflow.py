"""The workspace gate reports docs-only changes and fails closed for code changes."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = (REPO / ".github/workflows/rust.yml").read_text()
CHECKS = ("fmt", "taplo", "typos", "machete", "clippy", "doc", "serial-build",
          "test", "test-no-default-features", "test-ignored", "keccak-simd", "aarch64")


def job(name):
    return re.split(r"\n  (?=\S)", WORKFLOW.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]


def run_body(section):
    return textwrap.dedent(section.split("        run: |\n", 1)[1])


class RustWorkflowTests(unittest.TestCase):
    def test_every_check_depends_on_path_detection_and_the_gate_waits_for_it(self):
        gate = job("test-workspace")
        needs = re.search(r"needs: \[(.*?)\]", gate).group(1).split(", ")
        self.assertEqual(set(needs), {"changes", *CHECKS})
        for name in CHECKS:
            section = job(name)
            self.assertIn("needs: changes", section)
            self.assertIn("if: needs.changes.outputs.required == 'true'", section)
            self.assertIn(f"needs.{name}.result", gate)

    def test_docs_only_and_source_paths(self):
        shell = run_body(job("changes"))
        fake_git = '''\
            git() {
              printf '%s\\n' "$*" >> "$GIT_CALLS"
              if [ "$1" = cat-file ]; then
                test "$BASE_VALID" = true
              else
                test "$DIFF_VALID" = true || return 1
                printf '%s' "$CHANGED_PATHS"
              fi
            }
            '''
        cases = [
            ("README.md\nspecs/falcon.md\nspecs/notes/plan.md\n", "base", "true", "true", False),
            ("README.md\ncrates/aerie-bench/build.rs\n", "base", "true", "true", True),
            ("docs/README.md\n", "base", "true", "true", True),
            ("Cargo.lock\n", "base", "true", "true", True),
            ("specs/scripts/falcon_transcript_grinding.py\n", "base", "true", "true", True),
            ("specs/evidence.json\n", "base", "true", "true", True),
            (".github/workflows/rust.yml\n", "base", "true", "true", True),
            ("README.md\n", "", "true", "true", True),
            ("README.md\n", "base", "false", "true", True),
            ("README.md\n", "base", "true", "false", True),
        ]
        for event in ("pull_request", "push"):
            for paths, baseline, valid, diff_valid, required in cases:
                with self.subTest(event=event, paths=paths, valid=valid, baseline=baseline):
                    with tempfile.TemporaryDirectory() as directory:
                        output = Path(directory) / "outputs"
                        calls = Path(directory) / "calls"
                        env = dict(os.environ, EVENT_NAME=event, BASE_SHA=baseline, HEAD_SHA="head",
                                   BASE_VALID=valid, DIFF_VALID=diff_valid, CHANGED_PATHS=paths,
                                   GIT_CALLS=str(calls), GITHUB_OUTPUT=str(output), RUNNER_TEMP=directory)
                        result = subprocess.run(["bash", "-c", textwrap.dedent(fake_git) + shell],
                                                env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(output.read_text().strip(),
                                         f"required={str(required).lower()}")
                        if baseline and valid == "true":
                            comparison = "base...head" if event == "pull_request" else "base..head"
                            self.assertIn(comparison, calls.read_text())

    def test_gate_rejects_failed_detection_and_skipped_or_failed_code_checks(self):
        shell = run_body(job("test-workspace"))
        cases = [
            ("success", "false", "skipped", 0),
            ("success", "true", "success", 0),
            ("success", "true", "skipped", 1),
            ("failure", "false", "skipped", 1),
            ("success", "", "success", 1),
        ]
        for name in CHECKS:
            for value in ("failure", "cancelled"):
                cases.extend(("success", required, {name: value}, 1)
                             for required in ("true", "false"))
        for changes, required, results, expected in cases:
            statuses = dict.fromkeys(CHECKS, results if isinstance(results, str) else "success")
            if isinstance(results, dict):
                statuses.update(results)
            rendered = re.sub(r"\$\{\{ needs\.([\w-]+)\.result \}\}",
                              lambda match: statuses[match.group(1)], shell)
            env = dict(os.environ, CHANGES=changes, REQUIRED=required)
            with self.subTest(changes=changes, required=required, results=results):
                result = subprocess.run(["bash", "-c", rendered], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
