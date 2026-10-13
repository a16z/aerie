//! The `staged!` boundary helper of `specs/techniques/profiling.md`
//! section 3: one invocation defines both an Instant-timed region
//! (accumulated into a `u128` nanosecond field) and an entered tracing
//! span of the same extent, so a Perfetto trace and a TSV row cannot
//! drift apart. Exported from this crate because it is the lowest
//! instrumented layer. `aerie-falcon` inherits it as an existing dependency.
//!
//! Span emission with no subscriber installed is a static-branch check;
//! the accumulator keeps the exact `+=` semantics of the hand-placed
//! regions it replaces (multi-region fields sum across invocations).

/// Times `$body` into the `u128` nanosecond accumulator `$ns` while an
/// info-level span named `$name` covers exactly the same region.
///
/// `$ns` must be a place expression (a `mut` local or field); `$body` may
/// use `?`, in which case the region's time is not recorded for the
/// erroring run (the span still closes via RAII).
#[macro_export]
macro_rules! staged {
    ($name:literal, $ns:expr, $body:expr) => {{
        let __staged_span = ::tracing::info_span!($name).entered();
        let __staged_start = ::std::time::Instant::now();
        let __staged_result = $body;
        $ns += __staged_start.elapsed().as_nanos();
        drop(__staged_span);
        __staged_result
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn staged_accumulates_and_returns_the_body_value() {
        let mut ns: u128 = 0;
        let a = crate::staged!("statement.test.stage", ns, {
            std::thread::sleep(std::time::Duration::from_millis(2));
            21 * 2
        });
        assert_eq!(a, 42);
        assert!(ns > 0);
        let before = ns;
        let b = crate::staged!("statement.test.stage", ns, 1);
        assert_eq!(b, 1);
        assert!(ns >= before, "second region accumulates, not overwrites");
    }

    #[test]
    fn staged_propagates_question_mark() {
        // The `?` propagates before the accumulation runs, so `ns` is
        // never read on the error path; that skip is the documented
        // contract under test.
        #[allow(unused_variables, unused_assignments)]
        fn inner() -> Result<u32, ()> {
            let mut ns: u128 = 0;
            let v = crate::staged!("statement.test.err", ns, Err::<u32, ()>(())?);
            Ok(v)
        }
        assert_eq!(inner(), Err(()));
    }
}
