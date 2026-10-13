use std::fmt::{Display, Formatter};

/// Errors returned while constructing or validating an Aerie input or witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AerieError {
    InvalidRecord(String),
    InvalidPublicPolynomial(String),
    InvalidWitness(String),
    IntegerOverflow(&'static str),
    Profile(String),
}

impl Display for AerieError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRecord(message) => write!(formatter, "invalid record: {message}"),
            Self::InvalidPublicPolynomial(message) => {
                write!(formatter, "invalid public polynomial: {message}")
            }
            Self::InvalidWitness(message) => write!(formatter, "invalid witness: {message}"),
            Self::IntegerOverflow(operation) => {
                write!(formatter, "integer overflow while computing {operation}")
            }
            Self::Profile(message) => write!(formatter, "Aerie profile error: {message}"),
        }
    }
}

impl std::error::Error for AerieError {}

/// Result type used by the Aerie reference crate.
pub type Result<T> = std::result::Result<T, AerieError>;
