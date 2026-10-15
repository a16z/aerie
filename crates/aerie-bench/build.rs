use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn main() {
    rerun_if_exists(Path::new("build.rs"));
    if env::var_os("CARGO_FEATURE_FALCON_BENCH").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=AERIE_BENCH_BUILD_REVISION");

    let rustc = env::var_os("RUSTC")
        .and_then(|compiler| Command::new(compiler).arg("--version").output().ok())
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .filter(|version| !version.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=AERIE_BENCH_RUSTC={}", rustc.trim());

    let manifest = env::var_os("CARGO_MANIFEST_DIR");
    let root = manifest
        .as_deref()
        .map(Path::new)
        .and_then(Path::parent)
        .and_then(Path::parent);
    let lock = root.and_then(|root| {
        let lock = root.join("Cargo.lock");
        rerun_if_exists(&lock);
        fs::read_to_string(lock).ok()
    });
    let revisions = lock
        .as_deref()
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            line.strip_prefix("source = \"git+https://github.com/LayerZero-Labs/akita.git?")
                .and_then(|source| source.strip_suffix('"'))
                .and_then(|source| source.rsplit_once('#'))
                .map(|(_, revision)| revision)
                .filter(|revision| !revision.is_empty())
        })
        .collect::<BTreeSet<_>>();
    let akita_revision = if revisions.len() == 1 {
        revisions.first().copied().unwrap_or("unknown")
    } else {
        "unknown"
    };
    println!("cargo:rustc-env=AERIE_BENCH_AKITA_REV={akita_revision}");

    if let Some(root) = root {
        let branch = git(root, &["symbolic-ref", "-q", "HEAD"]).unwrap_or_default();
        for name in ["HEAD", "index", "packed-refs", branch.trim()] {
            if name.is_empty() {
                continue;
            }
            if let Some(path) = git(root, &["rev-parse", "--git-path", name]) {
                if !path.trim().is_empty() {
                    rerun_if_exists(&root.join(path.trim()));
                }
            }
        }
    }
    let mut revision = env::var("AERIE_BENCH_BUILD_REVISION")
        .ok()
        .filter(|revision| !revision.trim().is_empty())
        .or_else(|| root.and_then(|root| git(root, &["rev-parse", "HEAD"])))
        .map(|revision| revision.trim().to_owned())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    // Dirty means tracked changes only. Untracked fixtures and profiling output
    // do not affect the identity; Git index/ref changes refresh this probe.
    if let Some(status) =
        root.and_then(|root| git(root, &["status", "--porcelain", "--untracked-files=no"]))
    {
        if !status.is_empty() && revision != "unknown" && !revision.ends_with("-dirty") {
            revision.push_str("-dirty");
        }
    }
    println!("cargo:rustc-env=AERIE_BENCH_BUILD_REVISION={revision}");
}
