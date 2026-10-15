//! Exact integer semantics of the compact Falcon witness. This is shared
//! by witness preparation and adversarial oracles; it does not use a verifier.

use aerie_falcon_native::{params::FALCON_Q, Falcon1024, Falcon512, FalconParams};

use super::{
    check_falcon_v1_witness_ranges, check_range, supported_degree, CompactDirectBatch, DirectError,
    DirectParams,
};

impl CompactDirectBatch {
    /// Check lengths and canonical padding before accessing any row slices.
    pub fn validate_shape(&self) -> Result<(), DirectError> {
        let degree = supported_degree(self.degree())?;
        let cells = self.padded_rows.checked_mul(degree);
        if self.live_rows == 0
            || self.live_rows.checked_next_power_of_two() != Some(self.padded_rows)
            || cells != Some(self.h.len())
            || cells != Some(self.t.len())
            || cells != Some(self.s1.len())
            || cells != Some(self.s2.len())
            || cells != Some(self.v.len())
            || self.padded_rows.checked_mul(4) != Some(self.slack.len())
        {
            return Err(DirectError::InvalidLength);
        }
        Ok(())
    }

    /// Check the exact negacyclic ring identity, completed norm, derived
    /// ranges, encodability budget, and zero semantic padding. Source tables
    /// and projection outputs are checked by their protocol-layer owners.
    pub fn validate_semantics(&self) -> Result<(), DirectError> {
        self.validate_shape()?;
        match self.degree() {
            Falcon512::DEGREE => self.validate_semantics_of::<Falcon512>(),
            Falcon1024::DEGREE => self.validate_semantics_of::<Falcon1024>(),
            _ => Err(DirectError::InvalidLength),
        }
    }

    fn validate_semantics_of<P: DirectParams>(&self) -> Result<(), DirectError> {
        let degree = P::DEGREE;
        for (family, values, start) in [
            ("h-padding", &self.h, self.live_rows * degree),
            ("t-padding", &self.t, self.live_rows * degree),
            ("s1-padding", &self.s1, self.live_rows * degree),
            ("s2-padding", &self.s2, self.live_rows * degree),
            ("slack-padding", &self.slack, self.live_rows * 4),
        ] {
            check_range(family, values[start..].iter().copied().map(i64::from), 0, 0)?;
        }
        check_range(
            "v-padding",
            self.v[self.live_rows * degree..]
                .iter()
                .copied()
                .map(i64::from),
            0,
            0,
        )?;
        let validate_row = |row| {
            let start = row * degree;
            let end = start + degree;
            let h = &self.h[start..end];
            let s1 = &self.s1[start..end];
            let s2 = &self.s2[start..end];
            let t = &self.t[start..end];
            let v = self.v[start..end]
                .iter()
                .map(|&value| i64::from(value))
                .collect::<Vec<_>>();
            let slack = self.slack[row * 4..row * 4 + 4]
                .try_into()
                .expect("checked shape");
            let product = super::convolution_negacyclic_i16::<P>(h, s2);
            validate_row_semantics::<P>(t, s1, s2, &v, slack, &product)
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..self.live_rows)
                .into_par_iter()
                .try_for_each(validate_row)
        }
        #[cfg(not(feature = "parallel"))]
        (0..self.live_rows).try_for_each(validate_row)
    }
}

/// The product is an exact integer convolution, also available during normal
/// preparation. Even full i16 inputs contribute at most `n * 2^30`; the i32
/// quotient times q and the sum of `n + 4` squared i16 cells also fit i64/u64.
pub(super) fn validate_row_semantics<P: DirectParams>(
    t: &[i16],
    s1: &[i16],
    s2: &[i16],
    v: &[i64],
    slack: [i16; 4],
    product: &[i64],
) -> Result<(), DirectError> {
    let degree = P::DEGREE;
    if [t.len(), s1.len(), s2.len(), v.len(), product.len()] != [degree; 5] {
        return Err(DirectError::InvalidLength);
    }
    for (family, values) in [("s1", s1), ("slack", slack.as_slice())] {
        check_range(
            family,
            values.iter().copied().map(i64::from),
            -P::NORM_COEFFICIENT_ABS_BOUND,
            P::NORM_COEFFICIENT_ABS_BOUND,
        )?;
    }
    check_range(
        "v",
        v.iter().copied(),
        -P::QUOTIENT_ABS_BOUND,
        P::QUOTIENT_ABS_BOUND,
    )?;
    check_falcon_v1_witness_ranges::<P>(s1, s2, v, slack)?;
    for index in 0..degree {
        if product[index] + i64::from(s1[index]) - i64::from(t[index]) - FALCON_Q * v[index] != 0 {
            return Err(DirectError::RelationFailure(index));
        }
    }
    let completed_norm = s1
        .iter()
        .chain(s2)
        .chain(&slack)
        .map(|&cell| i64::from(cell).unsigned_abs().pow(2))
        .sum();
    if completed_norm != P::BETA_SQUARED {
        return Err(DirectError::NormMismatch {
            expected: P::BETA_SQUARED,
            actual: completed_norm,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_falcon_native::Falcon512;

    use crate::DirectRow;

    fn fixture() -> CompactDirectBatch {
        CompactDirectBatch::from_rows(
            &(0..3)
                .map(|seed| DirectRow::<Falcon512>::synthetic(seed + 263).unwrap())
                .collect::<Vec<_>>(),
        )
    }

    fn wide_fixture() -> CompactDirectBatch {
        CompactDirectBatch::from_rows(
            &(0..3)
                .map(|seed| DirectRow::<Falcon1024>::synthetic(seed + 263).unwrap())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn valid_and_equivalent_witnesses_pass() {
        for mut batch in [fixture(), wide_fixture()] {
            batch.validate_semantics().unwrap();
            // Slack witnesses need not be the constructor's canonical four squares.
            batch.slack[0] = -batch.slack[0];
            batch.slack.swap(1, 2);
            batch.validate_semantics().unwrap();
        }
    }

    #[test]
    fn in_range_ring_and_exact_norm_failures_are_detected() {
        for batch in [fixture(), wide_fixture()] {
            for column in ["s1", "s2", "v"] {
                let mut changed = batch.clone();
                match column {
                    "s1" => changed.s1[9] += 1,
                    "s2" => changed.s2[9] += 1,
                    _ => changed.v[9] += 1,
                }
                assert!(
                    matches!(
                        changed.validate_semantics(),
                        Err(DirectError::RelationFailure(_))
                    ),
                    "{column}"
                );
            }
            for delta in [-1, 1] {
                let mut changed = batch.clone();
                // Keep this norm mutation below the shared carrier cap.
                changed.slack[3] += delta;
                assert!(matches!(
                    changed.validate_semantics(),
                    Err(DirectError::NormMismatch { .. })
                ));
            }
        }
    }

    #[test]
    fn ring_convolution_is_negacyclic() {
        let mut h = [0; Falcon512::DEGREE];
        let mut s2 = [0; Falcon512::DEGREE];
        let mut t = [0; Falcon512::DEGREE];
        h[Falcon512::DEGREE - 1] = 1;
        s2[1] = 1;
        t[0] = -1;
        let row = DirectRow::<Falcon512>::new(h, t, [0; Falcon512::DEGREE], s2).unwrap();
        let mut batch = CompactDirectBatch::from_rows(&[row]);
        batch.validate_semantics().unwrap();
        batch.t[0] = 1;
        assert_eq!(
            batch.validate_semantics(),
            Err(DirectError::RelationFailure(0))
        );

        let mut h = [0; 1024];
        let mut s2 = [0; 1024];
        let mut t = [0; 1024];
        h[1023] = 1;
        s2[1] = 1;
        t[0] = -1;
        let row = DirectRow::<Falcon1024>::new(h, t, [0; 1024], s2).unwrap();
        let mut batch = CompactDirectBatch::from_rows(&[row]);
        batch.validate_semantics().unwrap();
        batch.t[0] = 1;
        assert_eq!(
            batch.validate_semantics(),
            Err(DirectError::RelationFailure(0))
        );
    }

    #[test]
    fn encodability_and_derived_bounds_are_checked() {
        let mut batch = fixture();
        batch.s2[..Falcon512::DEGREE].fill(1024);
        assert_eq!(
            batch.validate_semantics(),
            Err(DirectError::EncodingBudgetExceeded(4096))
        );
        for sign in [-1, 1] {
            for (family, value) in [
                ("s1", 5834),
                ("slack", 5834),
                ("v", 65999),
                ("s2-encodable", 2048),
            ] {
                let mut batch = fixture();
                let value = value * sign;
                match family {
                    "s1" => batch.s1[0] = value as i16,
                    "slack" => batch.slack[0] = value as i16,
                    "v" => batch.v[0] = value,
                    _ => batch.s2[0] = value as i16,
                }
                assert!(
                    matches!(batch.validate_semantics(), Err(DirectError::RangeViolation { family: actual, .. }) if actual == family)
                );
            }
        }
    }

    #[test]
    fn falcon_1024_derived_bounds_are_the_wider_ones() {
        let mut batch = wide_fixture();
        batch.s2[..1024].fill(1024);
        assert_eq!(
            batch.validate_semantics(),
            Err(DirectError::EncodingBudgetExceeded(8192))
        );
        for (family, value) in [("s1", 8383), ("slack", 8383), ("v", 134_110)] {
            let mut batch = wide_fixture();
            match family {
                "s1" => batch.s1[0] = value as i16,
                "slack" => batch.slack[0] = value as i16,
                _ => batch.v[0] = value,
            }
            assert!(
                matches!(batch.validate_semantics(), Err(DirectError::RangeViolation { family: actual, .. }) if actual == family),
                "{family}"
            );
        }
        // Values that Falcon-512 rejects are only relation failures here.
        let mut batch = wide_fixture();
        batch.s1[0] = 6000;
        assert!(matches!(
            batch.validate_semantics(),
            Err(DirectError::RelationFailure(0))
        ));
    }

    #[test]
    fn malformed_shapes_and_nonzero_padding_are_rejected() {
        let batch = fixture();
        for family in ["h", "t", "s1", "s2", "v", "slack"] {
            let mut changed = batch.clone();
            match family {
                "h" => changed.h[3 * Falcon512::DEGREE] = 1,
                "t" => changed.t[3 * Falcon512::DEGREE] = 1,
                "s1" => changed.s1[3 * Falcon512::DEGREE] = 1,
                "s2" => changed.s2[3 * Falcon512::DEGREE] = 1,
                "v" => changed.v[3 * Falcon512::DEGREE] = 1,
                _ => changed.slack[12] = 1,
            }
            assert!(
                matches!(
                    changed.validate_semantics(),
                    Err(DirectError::RangeViolation { .. })
                ),
                "{family}"
            );
        }
        let mut changed = batch.clone();
        changed.s2.pop();
        assert_eq!(
            changed.validate_semantics(),
            Err(DirectError::InvalidLength)
        );
        let mut changed = batch.clone();
        changed.live_rows = usize::MAX;
        assert_eq!(
            changed.validate_semantics(),
            Err(DirectError::InvalidLength)
        );
        // An unsupported degree (768 cells per row) is not a Falcon batch.
        let mut changed = batch;
        changed.h.resize(4 * 768, 0);
        changed.t.resize(4 * 768, 0);
        changed.s1.resize(4 * 768, 0);
        changed.s2.resize(4 * 768, 0);
        changed.v.resize(4 * 768, 0);
        assert_eq!(changed.validate_shape(), Err(DirectError::InvalidLength));
    }
}
