//! Falcon byte handling and native verification for Aerie.
//!
//! This crate owns the canonical-byte trust boundary: canonical public-key
//! and padded-signature decoding, direct-record construction, `HashToPoint`
//! preprocessing, native verification, and the encodability
//! witnesses derived from verified signatures. The decoded arithmetic types
//! expose read-only accessors, but their constructors are crate-private: the
//! public paths to verified arithmetic values are the single-record and
//! batched functions in [`witness`].
//!
//! Every byte-layer and native-verification type is generic over a sealed
//! [`FalconParams`] parameter set, [`Falcon512`] or [`Falcon1024`]. Select the
//! parameter set explicitly, for example `DirectRecord<Falcon512>` or
//! `DirectRecord<Falcon1024>`.

pub mod arithmetic;
pub mod encodability;
pub mod error;
mod falcon_ntt;
pub mod params;
pub mod preprocess;
pub mod record;
pub mod source;
pub mod witness;

pub use error::{AerieError, Result};
pub use params::{Falcon1024, Falcon512, FalconParams};
