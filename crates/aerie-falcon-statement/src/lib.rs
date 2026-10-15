//! Falcon public statement layer, lane-independent.
//!
//! Canonical byte-bound public records and verifier-local ingestion.
//! Nothing here depends on witness construction or a proof field.

pub mod direct;
pub mod production;
pub mod timing;

pub use direct::{DirectError, DirectParams};
pub use production::{
    AggregationRecord, AggregationStatement, ProductionError, MAX_PRODUCTION_BATCH_SIZE,
};
