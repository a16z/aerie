import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIMENSION_CODED_FAMILY = re.compile(r"aerie_fp64_d\d+(?:_|\b)")


class ScheduleFamilyNameTests(unittest.TestCase):
    def test_artifact_names_match_family_names_without_dimension_tokens(self):
        artifacts = sorted((ROOT / "crates").rglob("*.aks"))
        self.assertTrue(artifacts, "expected checked-in schedule artifacts")

        for artifact in artifacts:
            with self.subTest(artifact=artifact.relative_to(ROOT)):
                payload = json.loads(artifact.read_text())
                family_name = payload["family_name"]
                self.assertEqual(artifact.stem, family_name)
                self.assertIsNone(DIMENSION_CODED_FAMILY.search(family_name))

    def test_live_configuration_does_not_restore_dimension_coded_families(self):
        paths = [
            *sorted((ROOT / "crates").rglob("*.rs")),
            *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
            *sorted((ROOT / ".github" / "workflows").glob("*.yaml")),
        ]

        offenders = []
        for path in paths:
            if DIMENSION_CODED_FAMILY.search(path.read_text(errors="replace")):
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
