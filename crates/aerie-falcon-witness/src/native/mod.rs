//! Validated Falcon witness arithmetic and canonical direct records.

pub mod arithmetic;
pub mod encodability;
mod falcon_ntt;
pub mod record;
pub mod target;
pub mod witness;

pub use aerie_falcon_native::{error, params, preprocess, source};
pub use target::HashToPointTarget;
