#![cfg(feature = "falcon-bench")]

#[path = "../benches/falcon/args.rs"]
mod args;

use aerie_falcon::{wire::SaltMode, FalconDegree};
use args::BenchArgs;

fn parse(arguments: &[&str]) -> Result<BenchArgs, String> {
    BenchArgs::parse(arguments.iter().map(std::ffi::OsString::from))
}

fn assert_error(arguments: &[&str], expected: &str) {
    let error = parse(arguments).err().expect("arguments must be rejected");
    assert_eq!(error, expected);
}

#[test]
fn accepts_positive_batch_sizes_and_defaults() {
    assert_eq!(parse(&["8", "16", "16384"]).unwrap().sizes, [8, 16, 16_384]);
    let defaults = parse(&[]).unwrap();
    assert_eq!(defaults.sizes, [8, 1024, 16_384]);
    assert_eq!(defaults.salt_mode, SaltMode::PerRecord);
    assert!(!defaults.generate_only);
    assert!(!defaults.regenerate);
    assert!(defaults.heap_phase.is_none());
    assert!(defaults.profile_dir.is_none());
}

#[test]
fn accepts_both_degrees() {
    assert_eq!(
        parse(&["8", "--degree=512"]).unwrap().degree,
        FalconDegree::Falcon512
    );
    assert_eq!(
        parse(&["8", "--degree=1024"]).unwrap().degree,
        FalconDegree::Falcon1024
    );
}

#[test]
fn rejects_unknown_flags_and_degree_without_equals() {
    assert_error(&["--unknown"], "unrecognized argument: \"--unknown\"");
    assert_error(&["--degree", "1024"], "unrecognized argument: \"--degree\"");
}

#[test]
fn rejects_zero_and_overflow_batch_sizes() {
    assert_error(&["0"], "batch sizes must be positive");
    let overflow = format!("{}0", usize::MAX);
    assert_error(&[&overflow], &format!("invalid batch size: {overflow:?}"));
}

#[cfg(feature = "profiling")]
#[test]
fn profile_requires_exactly_one_explicit_batch_size() {
    assert_error(&["--profile"], "--profile takes exactly one batch size");
    assert_error(
        &["8", "16", "--profile=traces"],
        "--profile takes exactly one batch size",
    );
    assert_eq!(parse(&["8", "--profile"]).unwrap().sizes, [8]);
}

#[cfg(feature = "profiling")]
#[test]
fn rejects_profile_with_fixture_generation_only() {
    assert_error(
        &["8", "--profile", "--generate-honest-fixtures-only"],
        "--profile cannot be combined with --generate-honest-fixtures-only",
    );
}

#[cfg(not(feature = "heap-profile"))]
#[test]
fn heap_profile_requires_its_feature() {
    assert_error(
        &["8", "--heap-profile=verify"],
        "--heap-profile requires the heap-profile feature",
    );
}
