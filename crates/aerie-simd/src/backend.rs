//! Process-wide instruction-set ceiling for the runtime-selected kernels.
//!
//! Each kernel family picks its widest implementation at or below the
//! ceiling that this CPU supports. The ceiling is the widest detected ISA.
//! With the `backend-override` feature (enabled only through aerie-falcon's
//! dev-dependency, hence only in invocations that build its test or bench
//! targets), `AERIE_SIMD_BACKEND=auto|scalar|neon|avx2|avx512` pins it so
//! that one test binary can exercise every variant. The override can only
//! lower the ceiling: it panics on a backend this CPU does not support.

use std::sync::OnceLock;

/// An instruction-set ceiling for the runtime-selected kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Backend {
    /// Portable scalar code only.
    Scalar,
    /// `AArch64` Advanced SIMD.
    Neon,
    /// x86-64 AVX2.
    Avx2,
    /// x86-64 AVX-512 (each kernel states the subsets it needs).
    Avx512,
}

impl Backend {
    /// The ceiling selected for this process: the widest detected ISA, or
    /// the `AERIE_SIMD_BACKEND` request when the test-only override is built.
    #[must_use]
    pub fn selected() -> Self {
        static SELECTED: OnceLock<Backend> = OnceLock::new();
        *SELECTED.get_or_init(select)
    }

    /// The widest ISA this CPU supports.
    #[must_use]
    pub fn detected() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Self::Neon;
        }
        Self::Scalar
    }

    /// Lower-case name, as accepted by `AERIE_SIMD_BACKEND`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
        }
    }
}

#[cfg(not(feature = "backend-override"))]
fn select() -> Backend {
    Backend::detected()
}

#[cfg(feature = "backend-override")]
fn select() -> Backend {
    let requested = std::env::var("AERIE_SIMD_BACKEND").ok();
    let backend = match requested.as_deref() {
        None | Some("auto") => return Backend::detected(),
        Some("scalar") => Backend::Scalar,
        Some("neon") => Backend::Neon,
        Some("avx2") => Backend::Avx2,
        Some("avx512") => Backend::Avx512,
        Some(other) => {
            panic!("AERIE_SIMD_BACKEND must be auto, scalar, neon, avx2, or avx512, not {other}")
        }
    };
    let supported = match backend {
        Backend::Scalar => true,
        Backend::Neon => Backend::detected() == Backend::Neon,
        Backend::Avx2 | Backend::Avx512 => {
            Backend::detected() >= backend && Backend::detected() != Backend::Neon
        }
    };
    assert!(
        supported,
        "AERIE_SIMD_BACKEND={} is not supported by this CPU",
        backend.name()
    );
    backend
}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn selected_backend_matches_override() {
        if let Ok(requested) = std::env::var("AERIE_SIMD_BACKEND") {
            assert!(requested == "auto" || cfg!(feature = "backend-override"));
            if requested != "auto" {
                assert_eq!(Backend::selected().name(), requested);
            }
        }
    }
}
