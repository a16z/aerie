//! Formula-derived PIOP field-site grinding allocation from Section 10.1.

use super::{FalconDegree, MAX_RECORDS, MIN_PADDED_COEFFICIENTS};

const P: u128 = (1_u128 << 64) - 59;
const K: u128 = P * P;
// `2^128 - K`, expressed without trying to represent `2^128` in a `u128`.
const K_DEFICIT: u128 = u128::MAX - K + 1;
const MAX_PIOP_DEBIT_NUMERATOR: u32 = 26;

const fn maximum_rounds(degree: FalconDegree) -> usize {
    let maximum_padded = MAX_RECORDS.next_power_of_two();
    let minimum_padded = MIN_PADDED_COEFFICIENTS / degree.degree();
    let padded = if maximum_padded < minimum_padded {
        minimum_padded
    } else {
        maximum_padded
    };
    degree.logn() + padded.trailing_zeros() as usize
}

const _: () = assert!(maximum_rounds(FalconDegree::Falcon512) <= MAX_PIOP_DEBIT_NUMERATOR as usize);
const _: () =
    assert!(maximum_rounds(FalconDegree::Falcon1024) <= MAX_PIOP_DEBIT_NUMERATOR as usize);

/// Work bits for every PIOP field-challenge family in one protocol shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PiopGrindingPlan {
    pub rounds: usize,
    pub mu: u32,
    pub tau: u32,
    pub gamma: [u32; 3],
    pub sumcheck: [u32; 3],
}

impl PiopGrindingPlan {
    /// Derive the Section 10.1 plan from the parameter set and padded row count.
    pub fn for_shape(
        degree: FalconDegree,
        padded_records: usize,
    ) -> Result<Self, crate::UnsupportedBatch> {
        if !padded_records.is_power_of_two()
            || degree.padded_records(padded_records)? != padded_records
        {
            return Err(crate::UnsupportedBatch::Shape {
                degree: degree.degree(),
                records: padded_records,
            });
        }

        let rounds = degree.logn() + padded_records.trailing_zeros() as usize;
        let gamma_widths = match degree {
            FalconDegree::Falcon512 => [3, 8, 12],
            FalconDegree::Falcon1024 => [4, 9, 12],
        };

        Ok(Self {
            rounds,
            mu: grind_bits(8),
            tau: grind_bits(
                u32::try_from(rounds).map_err(|_| crate::UnsupportedBatch::Shape {
                    degree: degree.degree(),
                    records: padded_records,
                })?,
            ),
            gamma: gamma_widths.map(grind_bits),
            sumcheck: [3, 5, 3].map(grind_bits),
        })
    }
}

/// Return the least `g` such that `2^g >= 2^128 L / (2^64 - 59)^2`.
fn grind_bits(debit_numerator: u32) -> u32 {
    assert!(debit_numerator != 0);

    // Write `2^128 = K + K_DEFICIT`. For all registered PIOP numerators,
    // `L * K_DEFICIT < K`, so the exact integer ceiling of `2^128 L / K`
    // is `L + 1`. This retains the script's exact-cardinality calculation;
    // in particular, powers of two require one more bit than they would if K
    // were incorrectly rounded up to `2^128`.
    assert!(u128::from(debit_numerator) * K_DEFICIT < K);
    (debit_numerator + 1).next_power_of_two().ilog2()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fmt::Write as _;
    use std::path::Path;
    use std::process::Command;

    use aerie_falcon_native::{Falcon1024, Falcon512};

    use super::*;
    use crate::protocol::pcs::FalconVerifierSetup;
    use crate::protocol::{AggregationParams, MAX_RECORDS};

    /// Run the specification script's `--piop-plan-json` mode.
    fn script_piop_plan() -> String {
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../specs/scripts/falcon_transcript_grinding.py");
        let output = Command::new("python3")
            .arg(script)
            .arg("--piop-plan-json")
            .output()
            .expect("run transcript-grinding specification script");
        assert!(
            output.status.success(),
            "specification script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("UTF-8 JSON")
    }

    /// Read one unsigned integer field from a flat JSON object.
    fn json_usize(object: &str, key: &str) -> usize {
        let marker = format!("\"{key}\":");
        let start = object.find(&marker).expect("JSON field") + marker.len();
        let digits: String = object[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().expect("unsigned JSON integer")
    }

    /// The `(degree, padded records)` pairs the script's work plan covers.
    fn planned_shapes(plan: &str) -> BTreeSet<(usize, usize)> {
        let rows = plan
            .trim()
            .strip_prefix("[{")
            .and_then(|rows| rows.strip_suffix("}]"))
            .expect("non-empty JSON array of objects");
        let shapes: Vec<_> = rows
            .split("},{")
            .map(|row| (json_usize(row, "degree"), json_usize(row, "n_sig")))
            .collect();
        let set: BTreeSet<_> = shapes.iter().copied().collect();
        assert_eq!(set.len(), shapes.len(), "duplicate planned shape");
        set
    }

    /// Padded shapes that `supports()` admits over every record count in
    /// `1..=MAX_RECORDS`, requiring each record count's admission to match
    /// the plan's coverage of its padded shape.
    fn supported_shapes<P: AggregationParams>(
        planned: &BTreeSet<(usize, usize)>,
    ) -> BTreeSet<(usize, usize)> {
        let degree = P::DEGREE_ID;
        assert!(!FalconVerifierSetup::<P>::supports(0));
        assert!(!FalconVerifierSetup::<P>::supports(MAX_RECORDS + 1));
        let mut admitted = BTreeSet::new();
        for records in 1..=MAX_RECORDS {
            let shape = (
                degree.degree(),
                degree
                    .padded_records(records)
                    .expect("record count in range"),
            );
            let supported = FalconVerifierSetup::<P>::supports(records);
            assert_eq!(
                supported,
                planned.contains(&shape),
                "Falcon-{} records={records} padded={}: supports()={supported}",
                shape.0,
                shape.1,
            );
            if supported {
                admitted.insert(shape);
            }
        }
        admitted
    }

    #[test]
    fn grinding_plan_covers_exactly_the_shapes_the_verifier_supports() {
        let planned = planned_shapes(&script_piop_plan());
        let mut admitted = supported_shapes::<Falcon512>(&planned);
        admitted.extend(supported_shapes::<Falcon1024>(&planned));
        assert_eq!(admitted, planned);
    }

    #[test]
    fn piop_plan_matches_spec_script_for_all_admitted_shapes() {
        let actual = script_piop_plan();
        let mut expected = String::from("[");
        let mut first = true;
        for (degree, records) in planned_shapes(&actual) {
            let degree = FalconDegree::from_degree(degree).expect("Falcon degree");
            let plan = PiopGrindingPlan::for_shape(degree, records).expect("admitted shape");
            if !first {
                expected.push(',');
            }
            first = false;
            write!(
                expected,
                "{{\"degree\":{},\"gamma\":{:?},\"mu\":{},\"n_sig\":{},\"rounds\":{},\"sumcheck\":{:?},\"tau\":{}}}",
                degree.degree(),
                plan.gamma,
                plan.mu,
                records,
                plan.rounds,
                plan.sumcheck,
                plan.tau,
            )
            .expect("write expected JSON");
        }
        expected.push_str("]\n");
        let expected = expected.replace(' ', "");
        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_cardinality_adds_a_bit_at_power_of_two_debits() {
        assert_eq!(grind_bits(1), 1);
        assert_eq!(grind_bits(3), 2);
        assert_eq!(grind_bits(8), 4);
        assert_eq!(grind_bits(16), 5);
        assert_eq!(grind_bits(25), 5);
    }
}
