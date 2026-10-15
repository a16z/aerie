//! Execution helpers shared by the invariants: Rayon configuration and
//! panic traps for proving, verifying, and totality checks.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

static WORKER_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Configure Rayon before the first invariant check. Callers that do not call
/// it retain Rayon's machine-wide default.
pub fn configure_worker_threads(threads: usize) {
    WORKER_THREADS.store(threads.max(1), Ordering::Relaxed);
}

/// Configure Rayon's global pool with the requested worker count, once.
/// The configuration is a no-op if the pool was initialized elsewhere.
pub(crate) fn ensure_thread_pool() {
    static POOL: Once = Once::new();
    POOL.call_once(|| {
        let mut builder = rayon::ThreadPoolBuilder::new();
        let threads = WORKER_THREADS.load(Ordering::Relaxed);
        if threads != 0 {
            builder = builder.num_threads(threads);
        }
        let _ = builder.build_global();
    });
}

/// Run `body`, converting an unwinding panic to `Err(message)`.
/// Checks that deliberately probe for panics use [`trap_panic`] inside
/// `body` so they can classify the panic as a violation.
pub(crate) fn run_check<T, F>(body: F) -> Result<T, String>
where
    F: FnOnce() -> T,
{
    ensure_thread_pool();
    panic::catch_unwind(AssertUnwindSafe(body))
        .map_err(|payload| format!("check panicked: {}", panic_message(&*payload)))
}

/// Run `body`, catching any unwinding panic and returning it as `Err(message)`.
///
/// The installed host panic hook runs normally before the unwind is captured.
pub(crate) fn trap_panic<T, F>(body: F) -> Result<T, String>
where
    F: FnOnce() -> T,
{
    panic::catch_unwind(AssertUnwindSafe(body)).map_err(|payload| panic_message(&*payload))
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::trap_panic;
    use std::panic;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    const HOST_HOOK_CHILD: &str = "AERIE_INVARIANTS_HOST_HOOK_CHILD";

    #[test]
    fn trap_panic_returns_success() {
        assert_eq!(trap_panic(|| 42), Ok(42));
    }

    #[test]
    fn trap_panic_maps_string_payloads() {
        assert_eq!(
            trap_panic(|| panic!("borrowed panic")),
            Err("borrowed panic".to_owned())
        );
        assert_eq!(
            trap_panic(|| panic::panic_any(String::from("owned panic"))),
            Err("owned panic".to_owned())
        );
    }

    #[test]
    fn trap_panic_maps_non_string_payloads() {
        assert_eq!(
            trap_panic(|| panic::panic_any(42_u32)),
            Err("non-string panic payload".to_owned())
        );
    }

    #[test]
    fn trap_panic_calls_are_independent() {
        let barrier = Barrier::new(4);
        thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|index| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        assert_eq!(trap_panic(|| index), Ok(index));
                        let message = format!("panic from call {index}");
                        assert_eq!(
                            trap_panic(|| panic::panic_any(message.clone())),
                            Err(message)
                        );
                        assert_eq!(trap_panic(|| index + 1), Ok(index + 1));
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("independent panic trap call failed");
            }
        });
    }

    #[test]
    fn trap_panic_preserves_host_hook() {
        let output = Command::new(std::env::current_exe().expect("locate test executable"))
            .args([
                "--exact",
                "support::tests::trap_panic_host_hook_child",
                "--test-threads=1",
            ])
            .env(HOST_HOOK_CHILD, "1")
            .output()
            .expect("run host hook test in child process");
        assert!(
            output.status.success(),
            "host hook child failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }

    #[test]
    fn trap_panic_host_hook_child() {
        if std::env::var(HOST_HOOK_CHILD).as_deref() != Ok("1") {
            return;
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let hook_calls = Arc::clone(&calls);
        panic::set_hook(Box::new(move |_| {
            hook_calls.fetch_add(1, Ordering::SeqCst);
        }));

        assert_eq!(trap_panic(|| 42), Ok(42));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            trap_panic(|| panic!("first caught panic")),
            Err("first caught panic".to_owned())
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            trap_panic(|| panic::panic_any(42_u32)),
            Err("non-string panic payload".to_owned())
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        assert!(panic::catch_unwind(|| panic!("host panic after capture")).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}
