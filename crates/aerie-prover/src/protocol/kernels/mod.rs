//! Arithmetic kernels for the Falcon protocol stages.
//!
//! Field-arithmetic executors share the protocol's opening field. Transcript
//! ordering, proof types, wire encoding, and witness adaptation live in the
//! surrounding protocol modules.

pub(crate) mod kernel_common;
mod opening_field_ops;
mod quartic_relation;
mod range_source_lut;

pub(crate) mod product_kernel;
pub(crate) mod quartic_kernel;
pub(crate) mod range_source_kernel;

pub(crate) use kernel_common::{fold_rows_in_place, DigitSource, SignSource};
