use aerie_falcon::{wire::SaltMode, FalconDegree};

/// The phase a `--heap-profile` run records.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum HeapPhase {
    Prove,
    Verify,
    VerifierSetup,
}

pub(super) struct BenchArgs {
    pub(super) sizes: Vec<usize>,
    pub(super) degree: FalconDegree,
    pub(super) salt_mode: SaltMode,
    pub(super) generate_only: bool,
    pub(super) regenerate: bool,
    pub(super) heap_phase: Option<HeapPhase>,
    pub(super) profile_dir: Option<String>,
}

impl BenchArgs {
    pub(super) fn parse(
        arguments: impl Iterator<Item = std::ffi::OsString>,
    ) -> Result<Self, String> {
        let mut parsed = Self {
            sizes: Vec::new(),
            degree: FalconDegree::Falcon512,
            salt_mode: SaltMode::PerRecord,
            generate_only: false,
            regenerate: false,
            heap_phase: None,
            profile_dir: None,
        };
        for argument in arguments {
            let argument = argument
                .into_string()
                .map_err(|_| "arguments must be valid UTF-8".to_owned())?;
            match argument.as_str() {
                "--bench" => (),
                "--degree=512" => parsed.degree = FalconDegree::Falcon512,
                "--degree=1024" => parsed.degree = FalconDegree::Falcon1024,
                "--common-salt" => parsed.salt_mode = SaltMode::Common,
                "--generate-honest-fixtures-only" => parsed.generate_only = true,
                "--regenerate-honest-fixture" => parsed.regenerate = true,
                value if value == "--profile" || value.starts_with("--profile=") => {
                    if cfg!(not(feature = "profiling")) {
                        return Err("--profile requires the profiling feature".to_owned());
                    }
                    let directory = value.split_once('=').map_or("", |(_, directory)| directory);
                    parsed.profile_dir = Some(if directory.is_empty() {
                        "target/traces".to_owned()
                    } else {
                        directory.to_owned()
                    });
                }
                value if value.starts_with("--heap-profile=") => {
                    if cfg!(not(feature = "heap-profile")) {
                        return Err("--heap-profile requires the heap-profile feature".to_owned());
                    }
                    parsed.heap_phase = Some(match &value["--heap-profile=".len()..] {
                        "prove" => HeapPhase::Prove,
                        "verify" => HeapPhase::Verify,
                        "verifier-setup" => HeapPhase::VerifierSetup,
                        _ => {
                            return Err(
                                "--heap-profile expects prove, verify or verifier-setup".to_owned()
                            );
                        }
                    });
                }
                value if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) => {
                    let size = value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid batch size: {value:?}"))?;
                    if size == 0 {
                        return Err("batch sizes must be positive".to_owned());
                    }
                    parsed.sizes.push(size);
                }
                _ => return Err(format!("unrecognized argument: {argument:?}")),
            }
        }
        if parsed.profile_dir.is_some() {
            if parsed.sizes.len() != 1 {
                // The tracing subscriber can be installed only once per process.
                return Err("--profile takes exactly one batch size".to_owned());
            }
            if parsed.generate_only {
                return Err(
                    "--profile cannot be combined with --generate-honest-fixtures-only".to_owned(),
                );
            }
        }
        if parsed.sizes.is_empty() {
            parsed.sizes = vec![8, 1024, 16_384];
        }
        if parsed.heap_phase.is_some() && parsed.sizes.len() != 1 {
            // dhat allows one profiler per process.
            return Err("--heap-profile takes exactly one batch size".to_owned());
        }
        Ok(parsed)
    }
}
