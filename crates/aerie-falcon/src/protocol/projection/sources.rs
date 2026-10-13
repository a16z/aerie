//! The two projection sources (`specs/falcon.md` Section 6.1).
//!
//! - **Norm source `S`**: `2n` slots per record; slot `j + n a` holds
//!   `s_{1,i,j}` for `a = 0`, `slack_{i,j}` for `a = 1, j < 4`, and zero
//!   otherwise. Flat index `u = j + n a + 2n i`.
//! - **Residual source `Delta`**: `Delta_{i,j} = s_{1,i,j} + (h_i (*)
//!   s^*_{2,i})_j - t_{i,j}` with the negacyclic product over the integers. Flat
//!   index `w = j + n i`. Padded rows (`i >= live_rows`) have `h = t = 0`,
//!   so `Delta_i = s_{1,i}` there.
//!
//! `batch.s2` stores the `s_2^*` operand used by the negacyclic residual.

use aerie_falcon_native::{Falcon1024, Falcon512};
use aerie_falcon_statement::DirectParams;

use super::super::{FalconDegree, FalconWitness};

/// `Delta_i` of one record, exactly.
///
/// The integer carrier is `i64`. Every stored coefficient is an `i16`, so
/// `|s_1|, |t| <= 2^15` and each of the `n <= 2^10` products in `(h (*)
/// s_2^*)_j` has magnitude at most `2^30`. Hence `|Delta_{i,j}| <= 2^15 +
/// 2^40 + 2^15 < 2^41` for any batch contents, honest or not, and the
/// convolution's own `i64` output is exact
/// ([`aerie_falcon_statement::DirectParams`]'s exact CRT convolution).
pub(crate) fn delta_record(
    batch: &FalconWitness,
    degree: FalconDegree,
    record: usize,
    out: &mut [i64],
) {
    match degree {
        FalconDegree::Falcon512 => delta_record_of::<Falcon512>(batch, record, out),
        FalconDegree::Falcon1024 => delta_record_of::<Falcon1024>(batch, record, out),
    }
}

fn delta_record_of<P: DirectParams>(batch: &FalconWitness, record: usize, out: &mut [i64]) {
    let n = P::DEGREE;
    let range = record * n..(record + 1) * n;
    let s1 = &batch.s1[range.clone()];
    if record >= batch.live_rows {
        for (slot, &value) in out.iter_mut().zip(s1) {
            *slot = i64::from(value);
        }
        return;
    }
    let product = P::convolution_negacyclic_i16(&batch.h[range.clone()], &batch.s2[range.clone()]);
    let t = &batch.t[range];
    for (((slot, &s1), &product), &t) in out.iter_mut().zip(s1).zip(&product).zip(t) {
        *slot = i64::from(s1) + product - i64::from(t);
    }
}
