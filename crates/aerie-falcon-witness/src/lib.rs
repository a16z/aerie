//! Validated Falcon arithmetic, production witness construction, and fixtures.
//!
//! Public codecs and preprocessing belong to `aerie-falcon-native`; shared
//! statements and errors belong to `aerie-falcon-statement`.

pub mod direct;
#[cfg(any(test, feature = "test-fixtures"))]
pub mod fixtures;
#[cfg(feature = "honest-vectors")]
pub mod honest;
pub mod native;
pub mod production;

pub use aerie_falcon_statement::staged;
pub use aerie_falcon_statement::{
    AggregationRecord, AggregationStatement, DirectError, DirectParams, ProductionError,
    MAX_PRODUCTION_BATCH_SIZE,
};
pub use direct::{prepare_direct_witness_convolution, CompactDirectBatch, DirectRow};
pub use production::{AggregationWitness, FalconPreparedWitness, PreprocessingMetrics};

// Unsupported support operations for the combined protocol and invariant tests.
#[doc(hidden)]
pub use direct::{
    convolution_i16, convolution_negacyclic_i16, convolution_negacyclic_i16_from_mod_q,
};
