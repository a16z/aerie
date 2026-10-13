//! Execution helpers shared by the invariants: a deep-stack runner for the
//! prover/verifier kernels and a panic trap for totality checks.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

/// The prover and verifier recurse deeply enough that the default 8 MiB stack
/// overflows; the crate's own tests spawn 512 MiB threads for exactly this
/// reason. Every check that proves or verifies runs its body here.
const DEEP_STACK_BYTES: usize = 512 * 1024 * 1024;
static WORKER_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Configure Rayon before the first invariant check. Callers that do not call
/// it retain Rayon's machine-wide default.
pub fn configure_worker_threads(threads: usize) {
    WORKER_THREADS.store(threads.max(1), Ordering::Relaxed);
}

/// Size rayon's global worker threads to the prover stack, once.
///
/// The prover/verifier fan out onto rayon's global pool, whose workers ignore
/// our per-thread [`on_deep_stack`] size and would otherwise overflow their
/// default stack — the reason CI sets `RUST_MIN_STACK`. Building the pool here
/// makes the crate self-contained: no environment variable needed. It is a
/// no-op if the pool was already initialized elsewhere (in which case that
/// owner is responsible for the worker stack size).
pub(crate) fn ensure_deep_thread_pool() {
    static POOL: Once = Once::new();
    POOL.call_once(|| {
        let mut builder = rayon::ThreadPoolBuilder::new().stack_size(DEEP_STACK_BYTES);
        let threads = WORKER_THREADS.load(Ordering::Relaxed);
        if threads != 0 {
            builder = builder.num_threads(threads);
        }
        let _ = builder.build_global();
    });
}

/// Run `body` on a thread with a prover-sized stack and return its value.
///
/// If `body` itself panics (rather than returning), the panic is converted to
/// `Err(message)` so a harness bug never aborts the whole run. Checks that
/// deliberately probe for panics use [`trap_panic`] *inside* `body` instead, so
/// they can classify the panic as a violation.
pub fn on_deep_stack<T, F>(body: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    ensure_deep_thread_pool();
    let handle = std::thread::Builder::new()
        .stack_size(DEEP_STACK_BYTES)
        .spawn(body)
        .map_err(|error| format!("could not spawn worker thread: {error}"))?;
    handle
        .join()
        .map_err(|payload| format!("worker thread panicked: {}", panic_message(&*payload)))
}

/// Run `body`, catching any unwinding panic and returning it as `Err(message)`.
///
/// The default panic hook is silenced for the duration so a probed panic does
/// not spew a backtrace to stderr. This mutates the process-global hook, so
/// callers must not run `trap_panic` concurrently from multiple threads.
pub fn trap_panic<T, F>(body: F) -> Result<T, String>
where
    F: FnOnce() -> T,
{
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(body));
    panic::set_hook(previous);
    result.map_err(|payload| panic_message(&*payload))
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
