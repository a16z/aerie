//! Typed failures returned by aggregate setup, proving, and verification.

/// A failure in aggregate setup, statement ingestion, decoding, or verification.
#[derive(Debug)]
#[non_exhaustive]
pub enum AggregationError {
    /// The batch lies outside the supported setup or catalog geometry.
    UnsupportedBatch(UnsupportedBatch),
    /// Statement ingestion or witness construction failed.
    InvalidStatement(StatementError),
    /// Proof bytes or an in-memory proof cannot be canonically decoded or encoded.
    MalformedProof(MalformedProof),
    /// The proof or public inputs disagree with the statement or setup.
    StatementMismatch(StatementMismatch),
    /// A verification check rejected the proof at the indicated stage.
    ProofRejected { stage: RejectionStage },
    /// The honest prover's projection failed acceptance; this seed cannot prove the batch.
    ProjectionRejected,
    /// A commitment, opening, or internal prover operation failed.
    Backend(BackendError),
}

/// The first verification check that rejected the proof.
///
/// A wrong message, key, or salt diverges the transcript and surfaces as
/// `ProofRejected { stage: Grinding }` when the first subsequent proof-of-work check fails.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RejectionStage {
    /// The first-stage relation or projection acceptance check failed.
    Stage1,
    /// The second-stage relation failed.
    Stage2,
    /// The third-stage relation failed.
    Stage3,
    /// A proof-of-work check failed for the absorbed transcript.
    Grinding,
    /// The PCS opening failed verification or its backend could not decode the opening bytes.
    PcsOpening,
}

/// A batch outside the supported setup or catalog geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum UnsupportedBatch {
    /// The batch record count is unsupported.
    RecordCount { records: usize },
    /// The Falcon polynomial degree is unsupported.
    Degree { degree: usize },
    /// The parameter-set identifier is unsupported.
    ParameterSet { parameter_set: u8 },
    /// The degree and record count have no supported catalog entry.
    Shape { degree: usize, records: usize },
    /// The batch record count differs from the setup capacity.
    SetupSize { expected: usize, actual: usize },
    /// The batch does not support the selected salt mode.
    SaltMode,
}

/// Proof bytes, or an in-memory proof passed to the encoder, with invalid encoding or shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MalformedProof {
    /// The proof ends before a required value is complete.
    Truncated,
    /// Bytes remain after the complete proof or work stream.
    TrailingBytes,
    /// The wire version is unknown.
    UnknownVersion { version: u16 },
    /// The encoded parameter-set identifier is unknown.
    UnknownParameterSet { parameter_set: u8 },
    /// The encoded record count is invalid.
    InvalidRecordCount { records: usize },
    /// A field element is outside its canonical encoding range.
    NonCanonicalFieldElement,
    /// An encoded length is not canonical.
    NonCanonicalLength,
    /// A value has a noncanonical encoding.
    NonCanonicalEncoding,
    /// An encoded length exceeds the representable range.
    LengthOverflow,
    /// A proof frame has an unexpected kind.
    FrameKind,
    /// The proof magic does not identify this format.
    Magic,
    /// A commitment has an invalid shape.
    CommitmentShape,
    /// A stage proof has invalid dimensions.
    StageShape,
    /// Projection outputs cannot be represented by their canonical encoding.
    ProjectionEncoding,
    /// A work nonce has an invalid encoding or range.
    NonceEncoding,
}

/// A decoded proof header or shape that disagrees with the statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StatementMismatch {
    /// The proof record count differs from the statement.
    RecordCount { expected: usize, actual: usize },
    /// The input parameter set differs from the statement or setup.
    ParameterSet { expected: u8, actual: u8 },
    /// The number of supplied salts differs from the required count.
    SaltCount { expected: usize, actual: usize },
    /// Public input dimensions disagree.
    PublicInputShape,
    /// Padded public input rows contain nonzero values.
    NonZeroPadding,
}

/// Statement ingestion or witness construction failed.
#[derive(Debug)]
pub struct StatementError {
    error: aerie_falcon_statement::ProductionError,
    record: Option<usize>,
}

/// An opaque commitment, opening, or internal prover failure.
#[derive(Debug)]
pub struct BackendError(Box<dyn std::error::Error + Send + Sync + 'static>);

impl StatementError {
    #[doc(hidden)]
    #[must_use]
    pub fn new(error: aerie_falcon_statement::ProductionError) -> Self {
        Self {
            error,
            record: None,
        }
    }

    /// The zero-based index of the failing record, when known.
    #[must_use]
    pub const fn record(&self) -> Option<usize> {
        self.record
    }

    #[doc(hidden)]
    #[must_use]
    pub fn at_record(mut self, record: usize) -> Self {
        self.record = Some(record);
        self
    }
}

impl BackendError {
    #[doc(hidden)]
    pub fn new(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self(Box::new(error))
    }
}

impl AggregationError {
    #[doc(hidden)]
    pub fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Backend(BackendError::new(error))
    }

    #[doc(hidden)]
    #[must_use]
    pub fn internal(message: &'static str) -> Self {
        Self::backend(InternalError(message))
    }
}

#[derive(Debug)]
struct InternalError(&'static str);

impl std::fmt::Display for InternalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for InternalError {}

impl std::fmt::Display for RejectionStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stage1 => "stage one",
            Self::Stage2 => "stage two",
            Self::Stage3 => "stage three",
            Self::Grinding => "grinding",
            Self::PcsOpening => "PCS opening",
        })
    }
}

impl std::fmt::Display for UnsupportedBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RecordCount { records } => write!(f, "unsupported record count {records}"),
            Self::Degree { degree } => write!(f, "unsupported degree {degree}"),
            Self::ParameterSet { parameter_set } => {
                write!(f, "unsupported parameter set {parameter_set}")
            }
            Self::Shape { degree, records } => write!(
                f,
                "unsupported shape with degree {degree} and {records} records"
            ),
            Self::SetupSize { expected, actual } => {
                write!(f, "setup has {expected} records but batch has {actual}")
            }
            Self::SaltMode => f.write_str("unsupported salt mode for this batch"),
        }
    }
}

impl std::fmt::Display for MalformedProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownVersion { version } => write!(f, "unknown wire version {version}"),
            Self::UnknownParameterSet { parameter_set } => {
                write!(f, "unknown parameter set {parameter_set}")
            }
            Self::InvalidRecordCount { records } => {
                write!(f, "invalid encoded record count {records}")
            }
            Self::Truncated => f.write_str("truncated proof"),
            Self::TrailingBytes => f.write_str("trailing proof bytes"),
            Self::NonCanonicalFieldElement => f.write_str("noncanonical field element"),
            Self::NonCanonicalLength => f.write_str("noncanonical length"),
            Self::NonCanonicalEncoding => f.write_str("noncanonical encoding"),
            Self::LengthOverflow => f.write_str("encoded length overflow"),
            Self::FrameKind => f.write_str("incorrect proof frame kind"),
            Self::Magic => f.write_str("unknown proof magic"),
            Self::CommitmentShape => f.write_str("invalid commitment shape"),
            Self::StageShape => f.write_str("invalid stage shape"),
            Self::ProjectionEncoding => f.write_str("invalid projection encoding"),
            Self::NonceEncoding => f.write_str("invalid nonce encoding"),
        }
    }
}

impl std::fmt::Display for StatementMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RecordCount { expected, actual } => {
                write!(f, "statement has {expected} records but proof has {actual}")
            }
            Self::ParameterSet { expected, actual } => write!(
                f,
                "statement parameter set {expected} differs from proof parameter set {actual}"
            ),
            Self::SaltCount { expected, actual } => write!(
                f,
                "statement requires {expected} salts but proof has {actual}"
            ),
            Self::PublicInputShape => f.write_str("public input shapes disagree"),
            Self::NonZeroPadding => f.write_str("padded public input rows must be zero"),
        }
    }
}

impl std::fmt::Display for StatementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid statement: ")?;
        if let Some(record) = self.record {
            write!(f, "record {record}: ")?;
        }
        self.error.fmt(f)
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("aggregation backend failed")
    }
}

impl std::fmt::Display for AggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBatch(e) => e.fmt(f),
            Self::InvalidStatement(e) => e.fmt(f),
            Self::MalformedProof(e) => e.fmt(f),
            Self::StatementMismatch(e) => e.fmt(f),
            Self::Backend(e) => e.fmt(f),
            Self::ProofRejected { stage } => write!(f, "proof rejected at {stage}"),
            Self::ProjectionRejected => f.write_str("projection rejected for this prover seed"),
        }
    }
}

impl std::error::Error for StatementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.error.source()
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

impl std::error::Error for UnsupportedBatch {}

impl std::error::Error for MalformedProof {}

impl std::error::Error for StatementMismatch {}

impl std::error::Error for AggregationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnsupportedBatch(e) => e.source(),
            Self::InvalidStatement(e) => e.source(),
            Self::MalformedProof(e) => e.source(),
            Self::StatementMismatch(e) => e.source(),
            Self::Backend(e) => e.source(),
            Self::ProofRejected { .. } | Self::ProjectionRejected => None,
        }
    }
}

impl From<aerie_falcon_statement::ProductionError> for AggregationError {
    fn from(error: aerie_falcon_statement::ProductionError) -> Self {
        Self::InvalidStatement(StatementError::new(error))
    }
}

/// Propagate prover-internal Akita failures; verifier code must explicitly map
/// Akita results to the appropriate proof rejection.
impl From<akita_error::AkitaError> for AggregationError {
    fn from(error: akita_error::AkitaError) -> Self {
        Self::backend(error)
    }
}

impl From<UnsupportedBatch> for AggregationError {
    fn from(error: UnsupportedBatch) -> Self {
        Self::UnsupportedBatch(error)
    }
}

impl From<MalformedProof> for AggregationError {
    fn from(error: MalformedProof) -> Self {
        Self::MalformedProof(error)
    }
}

impl From<StatementMismatch> for AggregationError {
    fn from(error: StatementMismatch) -> Self {
        Self::StatementMismatch(error)
    }
}

impl From<crate::protocol::transcript::TranscriptOrderError> for AggregationError {
    fn from(error: crate::protocol::transcript::TranscriptOrderError) -> Self {
        use crate::protocol::transcript::TranscriptOrderError;
        match error {
            TranscriptOrderError::InvalidWork { .. } => Self::ProofRejected {
                stage: RejectionStage::Grinding,
            },
            TranscriptOrderError::NonceEncoding => {
                Self::MalformedProof(MalformedProof::NonceEncoding)
            }
            TranscriptOrderError::TrailingBytes => {
                Self::MalformedProof(MalformedProof::TrailingBytes)
            }
            other => Self::backend(other),
        }
    }
}
