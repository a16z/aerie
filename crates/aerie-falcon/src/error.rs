//! Errors returned by aggregate setup, proving, and verification.

/// A rejected aggregate input or a failure in its statement or PCS backend.
#[derive(Debug)]
pub enum AggregationError {
    /// A protocol shape, encoding, or verification check failed.
    InvalidInput(String),
    /// Canonical statement ingestion or witness construction failed.
    Statement(aerie_falcon_statement::ProductionError),
    /// The Akita commitment or opening backend rejected the operation.
    Akita(akita_error::AkitaError),
    /// The PCS opening did not authenticate the reconstructed claims.
    PcsOpening(akita_error::AkitaError),
    /// A commitment could not be serialized or decoded.
    Serialization(akita_serialization::SerializationError),
}

impl std::fmt::Display for AggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Statement(error) => error.fmt(f),
            Self::Akita(error) => error.fmt(f),
            Self::PcsOpening(error) => write!(f, "v1 multi-group PCS verifier: {error}"),
            Self::Serialization(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for AggregationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInput(_) => None,
            Self::Statement(error) => Some(error),
            Self::Akita(error) | Self::PcsOpening(error) => Some(error),
            Self::Serialization(error) => Some(error),
        }
    }
}

impl From<&str> for AggregationError {
    fn from(message: &str) -> Self {
        Self::InvalidInput(message.to_owned())
    }
}
impl From<String> for AggregationError {
    fn from(message: String) -> Self {
        Self::InvalidInput(message)
    }
}
impl From<aerie_falcon_statement::ProductionError> for AggregationError {
    fn from(error: aerie_falcon_statement::ProductionError) -> Self {
        Self::Statement(error)
    }
}
impl From<akita_error::AkitaError> for AggregationError {
    fn from(error: akita_error::AkitaError) -> Self {
        Self::Akita(error)
    }
}
impl From<akita_serialization::SerializationError> for AggregationError {
    fn from(error: akita_serialization::SerializationError) -> Self {
        Self::Serialization(error)
    }
}

impl From<crate::protocol::wire::WireError> for AggregationError {
    fn from(error: crate::protocol::wire::WireError) -> Self {
        match error {
            crate::protocol::wire::WireError::Serialization(source) => Self::Serialization(source),
            other => Self::InvalidInput(other.to_string()),
        }
    }
}
impl From<crate::protocol::transcript::TranscriptOrderError> for AggregationError {
    fn from(error: crate::protocol::transcript::TranscriptOrderError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}
