use crate::{
    error::{AerieError, Result},
    falcon_ntt::negacyclic_product,
    params::{filled, FalconParams, FALCON_Q, S_CAP},
};

/// Public, canonically decoded Falcon ring values for one record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicRow<P: FalconParams> {
    h_centered: P::Array<i16>,
    t_centered: P::Array<i16>,
}

impl<P: FalconParams> PublicRow<P> {
    /// Low-level constructor for decoder-owned values.
    ///
    /// This is deliberately crate-private: accepting detached decoded arrays
    /// as a production input would bypass canonical-byte binding.
    pub(crate) fn new(h_centered: &P::Array<i16>, t_centered: &P::Array<i16>) -> Result<Self> {
        for (name, polynomial) in [("h", h_centered), ("t", t_centered)] {
            if let Some((index, value)) = polynomial
                .as_ref()
                .iter()
                .enumerate()
                .find(|(_, value)| i64::from(**value).unsigned_abs() > 6_144)
            {
                return Err(AerieError::InvalidPublicPolynomial(format!(
                    "{name}[{index}]={value} is not a centered Falcon residue"
                )));
            }
        }
        Ok(Self {
            h_centered: *h_centered,
            t_centered: *t_centered,
        })
    }

    #[must_use]
    pub const fn h_centered(&self) -> &P::Array<i16> {
        &self.h_centered
    }

    #[must_use]
    pub const fn t_centered(&self) -> &P::Array<i16> {
        &self.t_centered
    }
}

/// Minimal decoded Falcon witness checked by native verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FalconWitness<P: FalconParams> {
    s1: P::Array<i16>,
    s2: P::Array<i16>,
    norm_squared: u64,
}

impl<P: FalconParams> FalconWitness<P> {
    /// Reconstruct `s1 = centered_q(t - h*s2)` and enforce the native norm.
    pub(crate) fn derive(public: &PublicRow<P>, s2: &P::Array<i16>) -> Result<Self> {
        enforce_signed_domain("s2", s2.as_ref(), -S_CAP, S_CAP - 1)?;
        let product = negacyclic_product::<P>(&public.h_centered, s2);
        let mut s1 = filled::<P, i16>(0);
        for ((slot, target), product) in s1
            .as_mut()
            .iter_mut()
            .zip(public.t_centered.as_ref())
            .zip(product.as_ref())
        {
            *slot = center_mod_q(i32::from(*target) - i32::from(*product));
        }
        enforce_signed_domain("s1", s1.as_ref(), -S_CAP, S_CAP - 1)?;
        let norm_squared =
            s1.as_ref()
                .iter()
                .chain(s2.as_ref())
                .try_fold(0_u64, |sum, value| {
                    let term = i64::from(*value).unsigned_abs().pow(2);
                    sum.checked_add(term)
                        .ok_or(AerieError::IntegerOverflow("Falcon norm"))
                })?;
        if norm_squared > P::BETA_SQUARED {
            return Err(AerieError::InvalidWitness(format!(
                "Falcon norm {norm_squared} exceeds {}",
                P::BETA_SQUARED
            )));
        }
        Ok(Self {
            s1,
            s2: *s2,
            norm_squared,
        })
    }

    #[must_use]
    pub const fn s1(&self) -> &P::Array<i16> {
        &self.s1
    }

    #[must_use]
    pub const fn s2(&self) -> &P::Array<i16> {
        &self.s2
    }

    #[must_use]
    pub const fn norm_squared(&self) -> u64 {
        self.norm_squared
    }
}

#[inline]
pub(crate) fn center_mod_q(value: i32) -> i16 {
    let residue = value.rem_euclid(FALCON_Q as i32);
    let centered = if residue > (FALCON_Q as i32 - 1) / 2 {
        residue - FALCON_Q as i32
    } else {
        residue
    };
    centered as i16
}

fn enforce_signed_domain<T>(name: &str, values: &[T], minimum: i64, maximum: i64) -> Result<()>
where
    T: Copy + Into<i64>,
{
    if let Some((index, value)) = values
        .iter()
        .copied()
        .map(Into::into)
        .enumerate()
        .find(|(_, value)| *value < minimum || *value > maximum)
    {
        return Err(AerieError::InvalidWitness(format!(
            "{name}[{index}]={value} lies outside [{minimum}, {maximum}]"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Falcon512;
    use crate::params::{Falcon1024, FALCON_BETA_SQUARED, FALCON_DEGREE};

    fn zero_public() -> PublicRow<Falcon512> {
        PublicRow::<Falcon512>::new(&[0; FALCON_DEGREE], &[0; FALCON_DEGREE])
            .expect("zero public row")
    }

    #[test]
    fn centered_reduction_covers_coefficient_differences_and_i32_endpoints() {
        // All differences of two i16 coefficients, plus the full helper
        // domain endpoints. Range plus congruence uniquely fixes centering.
        for value in (-65_535..=65_535).chain([i32::MIN, i32::MAX]) {
            let centered = center_mod_q(value);
            assert!((-6_144..=6_144).contains(&centered));
            assert_eq!(
                (i64::from(centered) - i64::from(value)).rem_euclid(FALCON_Q),
                0
            );
        }
    }

    #[test]
    fn falcon_512_relation_uses_the_degree_512_ring() {
        // X^511 * 2X = 2X^512 = -2 in Z_q[X]/(X^512 + 1).
        let mut h = [0_i16; FALCON_DEGREE];
        h[FALCON_DEGREE - 1] = 1;
        let mut t = [0_i16; FALCON_DEGREE];
        t[0] = -2;
        let public = PublicRow::<Falcon512>::new(&h, &t).expect("public row");
        let mut s2 = [0_i16; FALCON_DEGREE];
        s2[1] = 2;
        let witness =
            FalconWitness::<Falcon512>::derive(&public, &s2).expect("valid Falcon relation");
        assert_eq!(witness.s1, [0; FALCON_DEGREE]);
        assert_eq!(witness.norm_squared(), 4);
    }

    #[test]
    fn native_norm_rejects_large_decoded_s2() {
        let public = zero_public();
        let mut s2 = [0_i16; FALCON_DEGREE];
        s2[0] = 8_192;
        assert!(FalconWitness::derive(&public, &s2).is_err());
    }

    #[test]
    fn falcon_1024_witness_uses_its_own_norm_bound() {
        // Sixteen coefficients of magnitude 2,047 have squared norm
        // 67,043,344: above the Falcon-512 bound and below the Falcon-1024 one.
        let zero_1024 = filled::<Falcon1024, i16>(0);
        let public = PublicRow::<Falcon1024>::new(&zero_1024, &zero_1024).expect("public row");
        let mut s2 = filled::<Falcon1024, i16>(0);
        for cell in s2.as_mut().iter_mut().take(16) {
            *cell = 2_047;
        }
        let witness = FalconWitness::<Falcon1024>::derive(&public, &s2).expect("within bound");
        assert_eq!(witness.norm_squared(), 16 * 2_047 * 2_047);
        assert!(witness.norm_squared() > FALCON_BETA_SQUARED);
        assert_eq!(witness.s1().as_ref().len(), 1024);
        assert!(witness.s1().as_ref().iter().all(|&value| value == 0));

        let mut too_large = s2;
        too_large.as_mut()[16] = 2_047;
        assert!(FalconWitness::<Falcon1024>::derive(&public, &too_large).is_err());
    }

    #[test]
    fn falcon_1024_relation_uses_the_degree_1024_ring() {
        // X^1023 * 2X = 2X^1024 = -2 in Z_q[X]/(X^1024 + 1).
        let mut h = filled::<Falcon1024, i16>(0);
        h.as_mut()[1_023] = 1;
        let mut t = filled::<Falcon1024, i16>(0);
        t.as_mut()[0] = -2;
        let public = PublicRow::<Falcon1024>::new(&h, &t).expect("public row");
        let mut s2 = filled::<Falcon1024, i16>(0);
        s2.as_mut()[1] = 2;
        let witness = FalconWitness::<Falcon1024>::derive(&public, &s2).expect("valid relation");
        assert!(witness.s1().as_ref().iter().all(|&value| value == 0));
        assert_eq!(witness.norm_squared(), 4);
    }
}
