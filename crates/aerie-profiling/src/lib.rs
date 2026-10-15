//! Profiling infrastructure for the aerie-bench harness, as specified in
//! `specs/techniques/profiling.md`: a process-global tracing subscriber
//! writing Chrome-format traces (loadable in Perfetto or
//! `chrome://tracing`), a background memory/CPU metrics monitor, and the
//! post-run rewrite that turns monitor samples into Perfetto counter
//! tracks.
//!
//! This crate is never a dependency of the proof crates. Only `aerie-bench`
//! consumes it, behind the off-by-default `profiling` feature, activated
//! by the `--profile` bench flag. A profiled run's timings include
//! instrumentation overhead and are not citable benchmark numbers.

pub mod monitor;
pub mod rewrite;

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing_chrome::{ChromeLayerBuilder, FlushGuard};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub use monitor::{MetricsMonitor, MIN_INTERVAL};

/// Default filter: `aerie_falcon_statement`, `aerie_falcon_witness`,
/// `aerie_falcon`, `aerie_prover`, and `aerie_profiling` at debug (stage
/// spans are info; `staged!` spans take the call-site target, so the v1
/// protocol needs both protocol targets listed), the monitor, and the seven
/// akita crates that emit spans or events at the pinned revision at trace.
/// Falcon's proving kernels and orchestration use `aerie_prover`; common replay
/// and public arithmetic use `aerie_falcon`.
/// Public native preprocessing and witness native verification stay at warn:
/// `aerie_falcon_native` and `aerie_falcon_witness::native` need explicit
/// directives because filters match target prefixes.
/// Overridable via [`PROFILE_ENV_FILTER`]. Ambient `RUST_LOG` is deliberately
/// ignored because developer tools commonly set it for their own logging.
pub const DEFAULT_ENV_FILTER: &str =
    "warn,aerie_falcon_statement=debug,aerie_falcon=debug,aerie_prover=debug,\
     aerie_profiling=debug,aerie_falcon_native=warn,aerie_falcon_witness=debug,aerie_falcon_witness::native=warn,akita_pcs=trace,akita_prover=trace,akita_cpu_backend=trace,\
     akita_setup=trace,akita_types=trace,akita_algebra=trace,akita_verifier=trace";

/// Optional complete override for the profiling subscriber's event filter.
pub const PROFILE_ENV_FILTER: &str = "AERIE_PROFILE_FILTER";

fn profile_filter_from_override(
    override_filter: Option<String>,
) -> Result<EnvFilter, InstallError> {
    let configured = override_filter.unwrap_or_else(|| DEFAULT_ENV_FILTER.to_owned());
    EnvFilter::try_new(configured).map_err(|error| InstallError::Filter(error.to_string()))
}

fn profile_filter() -> Result<EnvFilter, InstallError> {
    let override_filter = match std::env::var(PROFILE_ENV_FILTER) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(InstallError::Filter(error.to_string())),
    };
    profile_filter_from_override(override_filter)
}

/// Installs the global subscriber (chrome layer + env filter) and starts
/// the metrics monitor. Call once per process, before any proving work;
/// hold the returned guard past the end of the profiled region and then
/// call [`ProfilingGuard::finish`].
pub fn install(trace_path: &Path, interval: Duration) -> Result<ProfilingGuard, InstallError> {
    if let Some(parent) = trace_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
        }
    }
    // Create the file here rather than via ChromeLayerBuilder::file(),
    // which panics on failure (e.g. an unwritable directory) instead of
    // taking the error path.
    let trace_file = std::fs::File::create(trace_path).map_err(InstallError::Io)?;
    let (chrome_layer, flush_guard) = ChromeLayerBuilder::new()
        .writer(std::io::BufWriter::new(trace_file))
        .include_args(true)
        .build();
    let filter = profile_filter()?;
    if let Err(e) = tracing_subscriber::registry()
        .with(filter)
        .with(chrome_layer)
        .try_init()
    {
        // Join the already-spawned writer thread before reporting.
        drop(flush_guard);
        return Err(InstallError::Subscriber(e.to_string()));
    }
    let monitor = MetricsMonitor::start(interval);
    Ok(ProfilingGuard {
        flush: Some(flush_guard),
        monitor: Some(monitor),
        trace_path: trace_path.to_path_buf(),
        finished: false,
    })
}

/// Owns the profiling session: the chrome trace's flush guard and the
/// monitor. Dropping it (or calling [`Self::finish`]) stops the monitor,
/// flushes the trace, and rewrites monitor samples into counter tracks.
///
/// The harness must hold this in `main()` and drop it only after the
/// roundtrip thread is joined; a `process::exit` while the guard lives
/// truncates the trace.
#[must_use = "dropping the guard ends the profiling session"]
pub struct ProfilingGuard {
    flush: Option<FlushGuard>,
    monitor: Option<MetricsMonitor>,
    trace_path: PathBuf,
    finished: bool,
}

impl ProfilingGuard {
    /// Ends the session explicitly and returns the trace path.
    pub fn finish(mut self) -> io::Result<PathBuf> {
        self.finish_inner()?;
        Ok(self.trace_path.clone())
    }

    fn finish_inner(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if let Some(mut monitor) = self.monitor.take() {
            monitor.stop();
        }
        // Dropping the FlushGuard flushes and finalizes the trace file;
        // the rewrite must run strictly after.
        drop(self.flush.take());
        rewrite::rewrite_counters(&self.trace_path)?;
        ensure_trace_has_events(&self.trace_path)
    }
}

fn ensure_trace_has_events(path: &Path) -> io::Result<()> {
    let raw = std::fs::read_to_string(path)?;
    let trace: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let events = match &trace {
        serde_json::Value::Array(events) => Some(events),
        serde_json::Value::Object(object) => {
            object.get("traceEvents").and_then(|value| value.as_array())
        }
        _ => None,
    };
    if events.is_some_and(Vec::is_empty) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("profiling trace contains no events; check that {PROFILE_ENV_FILTER} enables the repository span targets"),
        ));
    }
    Ok(())
}

impl Drop for ProfilingGuard {
    fn drop(&mut self) {
        // Best-effort on the drop path; finish() reports errors.
        let _ = self.finish_inner();
    }
}

/// Installation failure: the trace directory, or a subscriber already set
/// for this process (the flag is single-shot per process by design).
#[derive(Debug)]
pub enum InstallError {
    Io(io::Error),
    Filter(String),
    Subscriber(String),
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "creating the trace directory failed: {e}"),
            Self::Filter(e) => write!(f, "invalid {PROFILE_ENV_FILTER}: {e}"),
            Self::Subscriber(e) => write!(f, "installing the global subscriber failed: {e}"),
        }
    }
}

impl std::error::Error for InstallError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn empty_trace_is_rejected_instead_of_reported_as_an_artifact() {
        let path =
            std::env::temp_dir().join(format!("aerie-empty-trace-{}.json", std::process::id()));
        fs::write(&path, "[]").unwrap();
        let error = ensure_trace_has_events(&path).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains(PROFILE_ENV_FILTER));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn profiling_filter_defaults_to_required_repository_targets() {
        let filter = profile_filter_from_override(None).unwrap().to_string();
        for target in [
            "aerie_falcon_statement",
            "aerie_falcon",
            "aerie_prover",
            "aerie_profiling",
            "aerie_falcon_witness",
        ] {
            assert!(filter
                .split(',')
                .any(|directive| directive == format!("{target}=debug")));
        }
        assert_eq!(filter.split(',').count(), 15);
        let subscriber =
            tracing_subscriber::registry().with(profile_filter_from_override(None).unwrap());
        tracing::subscriber::with_default(subscriber, || {
            assert!(tracing::enabled!(target: "aerie_falcon", tracing::Level::DEBUG));
            assert!(tracing::enabled!(target: "aerie_prover", tracing::Level::DEBUG));
            assert!(tracing::enabled!(target: "aerie_falcon_statement", tracing::Level::DEBUG));
            assert!(tracing::enabled!(target: "aerie_profiling::monitor", tracing::Level::DEBUG));
            assert!(
                tracing::enabled!(target: "aerie_falcon_native::witness", tracing::Level::WARN)
            );
            assert!(
                !tracing::enabled!(target: "aerie_falcon_native::witness", tracing::Level::INFO)
            );
            assert!(
                tracing::enabled!(target: "aerie_falcon_witness::production", tracing::Level::DEBUG)
            );
            assert!(
                tracing::enabled!(target: "aerie_falcon_witness::native::witness", tracing::Level::WARN)
            );
            assert!(
                !tracing::enabled!(target: "aerie_falcon_witness::native::witness", tracing::Level::INFO)
            );
        });
        assert!(filter.contains("akita_pcs=trace"));
    }

    #[test]
    fn dedicated_profiling_filter_is_validated() {
        assert_eq!(
            profile_filter_from_override(Some("aerie_falcon=trace".to_owned()))
                .unwrap()
                .to_string(),
            "aerie_falcon=trace"
        );
        assert!(profile_filter_from_override(Some("aerie_falcon=[".to_owned())).is_err());
    }
}
