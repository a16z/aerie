//! The machine-checkable invariant abstraction.
//!
//! An [`Invariant`] pairs a JSON-serializable counterexample [`Input`] with a
//! deterministic [`check`] that returns one of three property verdicts or a
//! separate checker error:
//!
//! - [`Outcome::Holds`] — the property held on this input.
//! - [`Outcome::Violated`] — a confirmed, machine-checked counterexample.
//! - [`Outcome::Invalid`] — the input is degenerate or out of scope and is
//!   skipped (a no-op tamper, an out-of-range index, an unsupported batch
//!   shape). Skipped inputs count as neither a pass nor a violation, so the
//!   red-team loop never "wins" with a vacuous input.
//! - [`CheckError`] — checker infrastructure failed, so no property verdict can
//!   be reported.
//!
//! This mirrors Jolt's `jolt-eval` `Invariant` trait but drops the proc-macro,
//! the `Arbitrary`/fuzz machinery, and `schemars`: a caller needs only a serde
//! round-trip and a human-readable briefing.
//!
//! [`Input`]: Invariant::Input
//! [`check`]: Invariant::check

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;

/// A confirmed counterexample: the invariant did not hold on the input.
#[derive(Debug, Clone)]
pub struct Violation {
    /// One-line statement of what broke.
    pub summary: String,
    /// Everything a reader needs to reproduce and understand it.
    pub detail: String,
}

impl Violation {
    pub fn new(summary: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: detail.into(),
        }
    }
}

/// The result of checking one candidate input.
#[derive(Debug)]
pub enum Outcome {
    /// The property held: no counterexample here.
    Holds,
    /// A machine-checked counterexample.
    Violated(Violation),
    /// The input was degenerate or out of scope; skip it (neither pass nor
    /// violation). The string explains why, for the attempt log.
    Invalid(String),
}

impl Outcome {
    /// Convenience constructor for a skipped input.
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::Invalid(reason.into())
    }
}

/// The checker could not adjudicate an otherwise submitted candidate because
/// its own infrastructure failed. This is deliberately separate from
/// [`Outcome::Invalid`], which is reserved for malformed, degenerate, or
/// out-of-scope *inputs*.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct CheckError(pub String);

impl CheckError {
    /// Construct a checker-infrastructure failure with useful context.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// The result of attempting to adjudicate one typed candidate.
pub type CheckResult = Result<Outcome, CheckError>;

/// The self-contained input snapshot recorded for replay together with its
/// checker result.
#[must_use = "the adjudication contains both the checker result and replayable input"]
#[derive(Debug)]
pub struct Adjudication {
    /// The exact replayable JSON snapshot corresponding to
    /// [`Invariant::check`], with file-backed fields resolved inline. If
    /// resolution fails before the check can run, this remains the raw
    /// submission for diagnostics.
    pub recorded_input: serde_json::Value,
    /// The property verdict or checker-infrastructure failure.
    pub result: CheckResult,
}

/// The agent-facing description of an invariant: what it asserts, what a
/// counterexample is, and the exact JSON shape of the [`Invariant::Input`].
#[derive(Debug, Clone)]
pub struct Briefing {
    /// Prose the red-team agent reads: the property, the counterexample
    /// definition, and any scope limits (fixed batch shape, resource caps).
    pub summary: String,
    /// A precise description of the input JSON: fields, variants, ranges.
    pub input_doc: String,
    /// Repo-relative path to the invariant's own source, so the agent can read
    /// the exact `check` that will validate its proposals. Set with `file!()`.
    pub source: String,
}

/// A machine-checkable security property.
///
/// Implementors own any expensive setup internally (typically behind a
/// `OnceLock`), so [`check`](Invariant::check) is a pure function of the input
/// from the caller's point of view and the trait stays object-safe once erased
/// to [`DynInvariant`].
pub trait Invariant: Send + Sync {
    /// The counterexample candidate. Serialized for the agent prompt and the
    /// attempt log; deserialized from the agent's proposal.
    type Input: Serialize + DeserializeOwned + Clone + Debug;

    /// Stable identifier used on the command line and in logs.
    const NAME: &'static str;

    /// Agent-facing prose: the property, what a counterexample is, and any
    /// scope limits (fixed batch shape, resource caps).
    fn summary(&self) -> String;

    /// A precise description of the input JSON: fields, variants, ranges.
    fn input_description(&self) -> String;

    /// Repo-relative path to this invariant's source, so the agent can read the
    /// exact [`check`](Invariant::check) that will judge its proposals.
    ///
    /// The default derives the sibling `invariants/<name>.rs` from this trait
    /// file's own location (`file!()` is lexical, so it always names
    /// `invariant.rs` regardless of the implementor) and `NAME` (with `-`
    /// mapped to `_`). Override only if an invariant's file is named off that
    /// convention.
    fn source(&self) -> String {
        std::path::Path::new(file!())
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("invariants")
            .join(format!("{}.rs", Self::NAME.replace('-', "_")))
            .to_string_lossy()
            .into_owned()
    }

    /// The agent-facing briefing. The default composes [`summary`],
    /// [`input_description`], and [`source`]; rarely overridden.
    ///
    /// [`summary`]: Invariant::summary
    /// [`input_description`]: Invariant::input_description
    /// [`source`]: Invariant::source
    fn briefing(&self) -> Briefing {
        Briefing {
            summary: self.summary(),
            input_doc: self.input_description(),
            source: self.source(),
        }
    }

    /// Benign example inputs. They double as prompt examples and as an
    /// initial self-audit pass; a seed that unexpectedly [`Violated`] is a
    /// genuine finding surfaced immediately.
    ///
    /// [`Violated`]: Outcome::Violated
    fn seed_corpus(&self) -> Vec<Self::Input> {
        Vec::new()
    }

    /// Deterministically decide whether `input` is a counterexample. Checker
    /// infrastructure failures must be returned as [`CheckError`], never
    /// disguised as an out-of-scope [`Outcome::Invalid`] input.
    fn check(&self, input: &Self::Input) -> CheckResult;
}

/// Object-safe erasure of [`Invariant`]: the [`Input`](Invariant::Input) type
/// is replaced by [`serde_json::Value`] so a heterogeneous registry can be a
/// `Vec<Box<dyn DynInvariant>>` without Jolt's hand-written dispatch enum.
pub trait DynInvariant: Send + Sync {
    fn name(&self) -> &'static str;
    fn briefing(&self) -> Briefing;
    fn seed_corpus_json(&self) -> Vec<serde_json::Value>;

    /// Resolve `input` to one immutable snapshot, check that snapshot, and
    /// return the JSON form that must be recorded alongside the result. The
    /// blanket implementation for [`Invariant`] parses the raw input exactly
    /// once.
    ///
    /// A byte field may be submitted by file path (see `crate::bytes::Bytes`);
    /// deserializing reads the file, and re-serializing writes the bytes back
    /// inline. The typed value used by [`Invariant::check`] and the serialized
    /// value returned here are therefore the same immutable snapshot, even if
    /// the source file changes while the check is running.
    ///
    /// A deserialization failure is returned as [`Invalid`]; a serialization
    /// failure is a [`CheckError`] because the typed input cannot be recorded
    /// for replay. Both retain the raw submission for diagnostics. In either
    /// case [`Invariant::check`] is not called, and the input is never
    /// deserialized a second time.
    ///
    /// [`Invalid`]: Outcome::Invalid
    fn resolve_and_adjudicate(&self, input: &serde_json::Value) -> Adjudication;
}

impl<I: Invariant> DynInvariant for I {
    fn name(&self) -> &'static str {
        I::NAME
    }

    fn briefing(&self) -> Briefing {
        <Self as Invariant>::briefing(self)
    }

    fn seed_corpus_json(&self) -> Vec<serde_json::Value> {
        self.seed_corpus()
            .iter()
            .filter_map(|input| serde_json::to_value(input).ok())
            .collect()
    }

    fn resolve_and_adjudicate(&self, input: &serde_json::Value) -> Adjudication {
        let parsed: I::Input = match serde_json::from_value(input.clone()) {
            Ok(parsed) => parsed,
            Err(error) => {
                return Adjudication {
                    recorded_input: input.clone(),
                    result: Ok(Outcome::Invalid(format!(
                        "input did not match the schema: {error}"
                    ))),
                };
            }
        };
        let recorded_input = match serde_json::to_value(&parsed) {
            Ok(recorded_input) => recorded_input,
            Err(error) => {
                return Adjudication {
                    recorded_input: input.clone(),
                    result: Err(CheckError::new(format!(
                        "input could not be serialized for replay: {error}"
                    ))),
                };
            }
        };
        Adjudication {
            recorded_input,
            result: self.check(&parsed),
        }
    }
}

/// Every registered invariant, in a stable order.
#[must_use]
pub fn registry() -> Vec<Box<dyn DynInvariant>> {
    vec![
        Box::new(crate::invariants::Completeness),
        Box::new(crate::invariants::WitnessSoundness::default()),
        Box::new(crate::invariants::StatementBinding),
        Box::new(crate::invariants::VerifierTotality),
    ]
}

/// Look up a registered invariant by [`Invariant::NAME`].
#[must_use]
pub fn lookup(name: &str) -> Option<Box<dyn DynInvariant>> {
    registry().into_iter().find(|inv| inv.name() == name)
}

#[cfg(test)]
mod tests {
    use super::{DynInvariant, Invariant};
    use crate::invariants::{Completeness, StatementBinding, VerifierTotality, WitnessSoundness};
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Deserialize)]
    struct Unrecordable;

    impl Serialize for Unrecordable {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(<S::Error as serde::ser::Error>::custom(
                "deliberate serialization failure",
            ))
        }
    }

    struct UnrecordableInvariant;

    impl Invariant for UnrecordableInvariant {
        type Input = Unrecordable;
        const NAME: &'static str = "unrecordable";

        fn summary(&self) -> String {
            String::new()
        }

        fn input_description(&self) -> String {
            String::new()
        }

        fn check(&self, _input: &Self::Input) -> super::CheckResult {
            panic!("an unrecordable input must not reach the checker")
        }
    }

    #[test]
    fn source_defaults_to_the_sibling_invariant_file() {
        for (source, name) in [
            (Completeness.source(), "completeness"),
            (WitnessSoundness::default().source(), "witness_soundness"),
            (StatementBinding.source(), "statement_binding"),
            (VerifierTotality.source(), "verifier_totality"),
        ] {
            assert_eq!(
                source,
                format!("crates/aerie-invariants/src/invariants/{name}.rs")
            );
        }
    }

    #[test]
    fn completeness_briefing_prefers_file_backed_bytes() {
        let briefing = Invariant::briefing(&Completeness);
        let input_doc = &briefing.input_doc;
        assert!(
            input_doc.contains(r#"{"statement": "statement.bin", "witness": "witness.bin"}"#),
            "the prompt must show a file-backed value for both byte fields"
        );
        assert!(
            input_doc.contains("do NOT retype them"),
            "the prompt must steer the agent away from large inline arrays"
        );
        assert!(
            input_doc.contains("path to a file"),
            "the prompt must document file-backed byte fields"
        );
        assert!(
            briefing.summary.contains("production"),
            "{}",
            briefing.summary
        );
        assert!(
            !briefing.summary.contains("fallback") && !briefing.summary.contains("falling back"),
            "completeness must not advertise a fixture fallback: {}",
            briefing.summary
        );
    }

    #[test]
    fn statement_briefings_advertise_the_catalog_backed_domain() {
        let supported = crate::invariants::supported_statement_record_counts();
        assert_eq!(supported, (1..=16).collect::<Vec<_>>());
        assert!(aerie_falcon::FalconAggregateSetup::<aerie_falcon::Falcon512>::supports(4096));
        assert!(
            !supported.contains(&4096),
            "the harness retains its resource cap"
        );
        for briefing in [
            Invariant::briefing(&Completeness),
            Invariant::briefing(&StatementBinding),
        ] {
            for text in [&briefing.summary, &briefing.input_doc] {
                assert!(text.contains(&format!("{supported:?}")), "{text}");
                assert!(!text.contains("1..=16"), "{text}");
            }
        }
    }

    #[test]
    fn resolve_and_adjudicate_inlines_a_file_backed_byte_field() {
        let path =
            std::env::temp_dir().join(format!("aerie-adjudicate-{}.bin", std::process::id()));
        std::fs::write(&path, [1_u8, 2, 3, 4]).expect("write");
        let by_path =
            serde_json::json!({ "records": 8, "bytes": { "file": path.display().to_string() } });

        let adjudication = VerifierTotality.resolve_and_adjudicate(&by_path);
        // The path is resolved to the literal bytes, so the record no longer
        // depends on the file — deleting it does not change the recorded form.
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            adjudication.recorded_input,
            serde_json::json!({ "records": 8, "bytes": [1, 2, 3, 4] })
        );

        // A malformed input has no typed form to resolve and is not
        // passed to the invariant's check.
        let raw = serde_json::json!({ "records": 8 });
        let adjudication = VerifierTotality.resolve_and_adjudicate(&raw);
        assert_eq!(adjudication.recorded_input, raw);
        assert!(matches!(
            adjudication.result,
            Ok(super::Outcome::Invalid(_))
        ));
    }

    #[test]
    fn resolve_and_adjudicate_treats_serialization_as_a_checker_error() {
        let raw = serde_json::Value::Null;
        let adjudication = UnrecordableInvariant.resolve_and_adjudicate(&raw);

        assert_eq!(adjudication.recorded_input, raw);
        let error = adjudication.result.unwrap_err();
        assert!(error
            .to_string()
            .contains("input could not be serialized for replay"));
    }
}
