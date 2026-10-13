//! Exact, runtime-selected CPU kernels for the Aerie Falcon protocol.
//!
//! Safe plans select an architecture implementation once, then reuse it in
//! hot loops. Architecture-specific unsafe code is isolated in private leaf
//! modules; protocol crates never manipulate vector registers or raw pointers.

mod backend;
mod falcon_unpack;
mod small_signed_dot;

pub use backend::Backend;
pub use falcon_unpack::KeyUnpackPlan;
pub use small_signed_dot::SmallSignedDotPlan;
