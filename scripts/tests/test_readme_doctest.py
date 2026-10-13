import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ReadmeDoctestTests(unittest.TestCase):
    def test_facade_example_matches_readme(self):
        readme = (ROOT / "README.md").read_text()
        readme_match = re.search(
            r"This example uses the `aerie` facade:\n\n```rust\n(.*?)\n```",
            readme,
            re.DOTALL,
        )
        self.assertIsNotNone(readme_match, "README facade example not found")

        lib = (ROOT / "crates" / "aerie" / "src" / "lib.rs").read_text()
        doctest_match = re.search(
            r"//! # Example\n//!\n//! ```no_run\n((?://!.*\n)+?)//! ```",
            lib,
        )
        self.assertIsNotNone(doctest_match, "facade no_run doctest not found")
        doctest = "\n".join(
            line.removeprefix("//! ") if line != "//!" else ""
            for line in doctest_match.group(1).splitlines()
        )

        self.assertEqual(readme_match.group(1), doctest)


if __name__ == "__main__":
    unittest.main()
