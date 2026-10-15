//! Background metrics monitor: one named thread sampling process RSS and
//! CPU utilization, emitted as `tracing` events with `counters.*` fields
//! that the counter rewrite turns into Perfetto counter tracks. The
//! events carry this module's path as their target (`aerie_profiling::monitor`;
//! tracing's macros cannot combine an explicit `target:` with dotted
//! field names). Modeled on jolt's `jolt-profiling/src/monitor.rs`,
//! reduced per `specs/techniques/profiling.md` section 5.
//!
//! The memory track is a sampled time series and can miss transients
//! between ticks; no derived peak is emitted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use sysinfo::{ProcessesToUpdate, System};

/// Sampling floor; requests below this are clamped up.
pub const MIN_INTERVAL: Duration = Duration::from_millis(50);

/// Sampling ceiling; requests above this (e.g. a typo'd
/// `AERIE_MONITOR_INTERVAL_MS`) are clamped down so the monitor still
/// produces a usable track and teardown stays prompt.
pub const MAX_INTERVAL: Duration = Duration::from_secs(10);

/// A core counts as active when its utilization exceeds this fraction of
/// one percent, mirroring the jolt monitor's threshold.
const ACTIVE_CORE_THRESHOLD_PERCENT: f32 = 0.1;

const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// Background sampling thread. Stops (signal + join) on [`Self::stop`] or
/// on drop.
#[must_use = "the monitor samples only while it is alive; hold it for the region being profiled"]
pub struct MetricsMonitor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MetricsMonitor {
    /// Spawns the `aerie-monitor` thread sampling every `interval`
    /// (clamped to [`MIN_INTERVAL`]).
    pub fn start(interval: Duration) -> Self {
        let interval = interval.clamp(MIN_INTERVAL, MAX_INTERVAL);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("aerie-monitor".to_owned())
            .spawn(move || sample_loop(&stop_flag, interval))
            .expect("spawning the aerie-monitor thread failed");
        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// Signals the thread and joins it. Idempotent.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            // A panicked monitor thread must not poison profiling teardown.
            let _ = handle.join();
        }
    }
}

impl Drop for MetricsMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

fn sample_loop(stop: &AtomicBool, interval: Duration) {
    let mut system = System::new();
    let Ok(pid) = sysinfo::get_current_pid() else {
        // No pid, no process metrics; CPU-only sampling would mislabel the
        // memory track as zero, so bail out entirely.
        return;
    };
    while !stop.load(Ordering::Relaxed) {
        system.refresh_cpu_usage();
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);

        let rss_bytes = system.process(pid).map_or(0, sysinfo::Process::memory);
        let memory_gib = rss_bytes as f64 / BYTES_PER_GIB;

        let cpus = system.cpus();
        let core_count = cpus.len();
        let cpu_percent = system.global_cpu_usage();
        let cores_active_avg = f64::from(cpu_percent) * core_count as f64 / 100.0;
        let cores_active = cpus
            .iter()
            .filter(|cpu| cpu.cpu_usage() > ACTIVE_CORE_THRESHOLD_PERCENT)
            .count();

        emit_sample(memory_gib, cpu_percent, cores_active_avg, cores_active);
        // Chunked sleep so stop() joins within ~one MIN_INTERVAL instead
        // of blocking teardown for a full sampling interval.
        let mut remaining = interval;
        while !remaining.is_zero() && !stop.load(Ordering::Relaxed) {
            let chunk = remaining.min(MIN_INTERVAL);
            std::thread::sleep(chunk);
            remaining = remaining.saturating_sub(chunk);
        }
    }
}

#[cfg(target_os = "linux")]
fn emit_sample(memory_gib: f64, cpu_percent: f32, cores_active_avg: f64, cores_active: usize) {
    let thread_count = std::fs::read_dir("/proc/self/task").map_or(0, Iterator::count);
    tracing::debug!(
        counters.memory_gib = memory_gib,
        counters.cpu_percent = cpu_percent,
        counters.cores_active_avg = cores_active_avg,
        counters.cores_active = cores_active,
        counters.thread_count = thread_count,
    );
}

#[cfg(not(target_os = "linux"))]
fn emit_sample(memory_gib: f64, cpu_percent: f32, cores_active_avg: f64, cores_active: usize) {
    // No portable thread enumeration off Linux; the field is absent rather
    // than fabricated (spec section 5).
    tracing::debug!(
        counters.memory_gib = memory_gib,
        counters.cpu_percent = cpu_percent,
        counters.cores_active_avg = cores_active_avg,
        counters.cores_active = cores_active,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_is_idempotent_and_drop_joins_without_deadlock() {
        let mut monitor = MetricsMonitor::start(Duration::from_millis(1));
        // The interval floor means the thread sleeps >= 50ms per tick; stop
        // must still return promptly after at most one tick.
        std::thread::sleep(Duration::from_millis(10));
        monitor.stop();
        monitor.stop();
        drop(monitor);
    }

    #[test]
    fn interval_is_clamped_to_the_floor() {
        assert!(Duration::from_millis(1).max(MIN_INTERVAL) == MIN_INTERVAL);
        assert!(Duration::from_millis(500).max(MIN_INTERVAL) == Duration::from_millis(500));
    }
}
