"""Build metadata probes must tolerate packaged layouts and unavailable tools."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]


class BuildMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (REPO / "target/build-metadata-tests").mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=REPO / "target/build-metadata-tests")
        cls.binary = Path(cls.directory.name) / "metadata"
        subprocess.run(["rustc", str(REPO / "crates/aerie-bench/build.rs"), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True)
        cls.rustc = shutil.which("rustc")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_probe(self, root, *, enabled=True, git=True, compiler=True):
        manifest = root / "crates/aerie-bench"
        manifest.mkdir(parents=True, exist_ok=True)
        (manifest / "build.rs").write_text("// build script\n")
        env = dict(os.environ, CARGO_MANIFEST_DIR=str(manifest))
        env.pop("AERIE_BENCH_BUILD_REVISION", None)
        env.pop("CARGO_FEATURE_FALCON_BENCH", None)
        if enabled:
            env["CARGO_FEATURE_FALCON_BENCH"] = "1"
        env["RUSTC"] = self.rustc if compiler else str(root / "missing-rustc")
        if not git:
            env["PATH"] = str(root / "missing-bin")
        return subprocess.run([str(self.binary)], cwd=manifest, env=env,
                              check=True, capture_output=True, text=True).stdout

    def test_feature_disabled_does_not_probe(self):
        with tempfile.TemporaryDirectory(dir=self.directory.name) as directory:
            output = self.run_probe(Path(directory), enabled=False, git=False, compiler=False)
            self.assertNotIn("cargo:rustc-env=", output)

    def test_missing_git_lock_and_compiler_are_unknown(self):
        with tempfile.TemporaryDirectory(dir=self.directory.name) as directory:
            output = self.run_probe(Path(directory), git=False, compiler=False)
            for name in ("RUSTC", "AKITA_REV", "BUILD_REVISION"):
                self.assertIn(f"cargo:rustc-env=AERIE_BENCH_{name}=unknown\n", output)
            self.assertNotIn("Cargo.lock", output)

    def test_lock_without_git_revisions_or_with_multiple_revisions_is_unknown(self):
        with tempfile.TemporaryDirectory(dir=self.directory.name) as directory:
            root = Path(directory)
            for content in (
                '[package]\nname = "akita-pcs"\n',
                '\n'.join(f'source = "git+https://github.com/LayerZero-Labs/akita.git?rev=x#{rev}"'
                          for rev in ("aaa", "bbb")),
            ):
                (root / "Cargo.lock").write_text(content)
                output = self.run_probe(root, git=False)
                self.assertIn("cargo:rustc-env=AERIE_BENCH_AKITA_REV=unknown\n", output)
                self.assertIn(f"cargo:rerun-if-changed={root / 'Cargo.lock'}\n", output)

    def test_packed_ref_registers_only_existing_identity_inputs(self):
        with tempfile.TemporaryDirectory(dir=self.directory.name) as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "README.md").write_text("tracked readme\n")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture",
                            "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"],
                           check=True)
            subprocess.run(["git", "-C", str(root), "pack-refs", "--all", "--prune"], check=True)
            output = self.run_probe(root)
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                               text=True).strip()
            self.assertIn(f"cargo:rustc-env=AERIE_BENCH_BUILD_REVISION={revision}\n", output)
            paths = [line.partition("=")[2] for line in output.splitlines()
                     if line.startswith("cargo:rerun-if-changed=")]
            self.assertTrue(all((root / "crates/aerie-bench" / path).exists() for path in paths))
            self.assertFalse(any("refs/heads/" in path or "README.md" in path for path in paths))
            self.assertTrue(any(path.endswith("packed-refs") for path in paths))
            (root / "README.md").write_text("tracked change\n")
            dirty_output = self.run_probe(root)
            self.assertIn(f"cargo:rustc-env=AERIE_BENCH_BUILD_REVISION={revision}-dirty\n",
                          dirty_output)


if __name__ == "__main__":
    unittest.main()
