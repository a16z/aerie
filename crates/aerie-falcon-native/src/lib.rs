//! Canonical Falcon byte handling and public preprocessing for Aerie.
//!
//! This crate owns canonical public-key and padded-signature decoding and
//! public `HashToPoint` preprocessing. Witness construction and validated
//! signature arithmetic belong to `aerie-falcon-witness`.
//!
//! The byte-layer types are generic over the sealed [`FalconParams`]
//! parameter set, [`Falcon512`] or [`Falcon1024`].

pub mod error;
pub mod params;
pub mod preprocess;
pub mod source;

pub use error::{AerieError, Result};
pub use params::{Falcon1024, Falcon512, FalconParams};
