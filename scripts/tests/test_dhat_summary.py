import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "dhat_summary.py"


def site(gb: int, tb: int, tbk: int, frames: list[int], eb: int = 0) -> dict:
    return {"gb": gb, "eb": eb, "tb": tb, "tbk": tbk, "fs": frames}


class DhatSummaryTests(unittest.TestCase):
    def test_groups_sites_by_innermost_workspace_frame(self):
        frames = [
            "[root]",
            "0x1: alloc::raw_vec::finish_grow::h0123456789abcdef (/rustc/abc/alloc/src/raw_vec.rs:10:5)",
            "0x2: aerie_falcon::protocol::instance::derive::h0123456789abcdef "
            "(/home/u/aerie/crates/aerie-falcon/src/protocol/instance.rs:55:21)",
            "0x3: aerie_falcon::protocol::transport::verify::h0123456789abcdef "
            "(/home/u/aerie/crates/aerie-falcon/src/protocol/transport.rs:364:9)",
            "0x4: std::rt::lang_start::h0123456789abcdef (/rustc/abc/std/src/rt.rs:1:1)",
        ]
        profile = {
            "ftbl": frames,
            "pps": [
                site(3 << 20, 3 << 20, 1, [1, 2, 3]),
                site(1 << 20, 1 << 20, 1, [1, 2, 3], eb=1 << 20),
                site(0, 5 << 20, 7, [1, 4]),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dhat.json"
            path.write_text(json.dumps(profile))
            output = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertIn("peak live heap 4.0 MiB; live at end 1.0 MiB; allocated 9.0 MiB in 9 blocks; 3 sites", output)
        peak = output.split("## Live at the global peak\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn(
            "4.0 MiB (100.0%)  aerie_falcon::protocol::instance::derive "
            "(crates/aerie-falcon/src/protocol/instance.rs:55)",
            peak,
        )
        self.assertIn(
            "4.0 MiB via aerie_falcon::protocol::transport::verify "
            "(crates/aerie-falcon/src/protocol/transport.rs:364)",
            peak,
        )
        end = output.split("## Live when the profile ended\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("1.0 MiB (100.0%)  aerie_falcon::protocol::instance::derive", end)
        count = output.split("## Allocation count\n", 1)[1]
        self.assertTrue(count.lstrip().startswith("- ") and "(no workspace frame)" in count.splitlines()[0])

    def test_rejects_malformed_profile_with_message(self):
        for text in ("{not json", '{"pps": []}', "[]"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "dhat.json"
                path.write_text(text)
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("not a dhat profile", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
