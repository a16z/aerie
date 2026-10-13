//! Falcon statement layer, lane-independent.
//!
//! Byte-bound records, production ingestion, and constructed fixtures.
//! Nothing here depends on a proof field.

pub mod direct;
pub mod fixtures;
#[cfg(feature = "honest-vectors")]
pub mod honest;
pub mod production;
pub mod timing;

pub use direct::{
    prepare_direct_witness_convolution, CompactDirectBatch, DirectBatch, DirectError, DirectParams,
    DirectPublicBatch, DirectRow, FALCON_BETA_SQUARED, FALCON_DEGREE, FALCON_Q,
};
pub use production::{
    split_source_statement, AggregationRecord, AggregationStatement, AggregationWitness,
    FalconPreparedWitness, PreprocessingMetrics, ProductionError, ProductionStatement,
    MAX_PRODUCTION_BATCH_SIZE,
};
