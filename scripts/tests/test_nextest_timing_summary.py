import json
import tempfile
import unittest
from pathlib import Path

from scripts.nextest_timing_summary import markdown, test_cases, wall_seconds


class NextestTimingSummaryTests(unittest.TestCase):
    def test_renders_slowest_tests_and_wall_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            junit = root / "junit.xml"
            junit.write_text(
                """<testsuites><testsuite>
                <testcase classname="crate::module" name="fast" time="0.25" />
                <testcase classname="crate::module" name="slow" time="2.5" />
                </testsuite></testsuites>""",
                encoding="utf-8",
            )
            timing = root / "timing.json"
            timing.write_text(
                json.dumps({"started_at_epoch": 10, "finished_at_epoch": 17}),
                encoding="utf-8",
            )

            cases = test_cases(junit)
            rendered = markdown(cases, wall_seconds(timing), limit=1)

            self.assertIn("2 tests; 2.75s summed test time; 7s job-step wall time", rendered)
            self.assertIn("crate::module::slow", rendered)
            self.assertNotIn("crate::module::fast", rendered)


if __name__ == "__main__":
    unittest.main()
