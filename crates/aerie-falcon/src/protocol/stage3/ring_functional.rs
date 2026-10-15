use crate::protocol::ring::PublicRingKeys;
use crate::substrate::OpeningField as E;
use jolt_field::Ring;

#[must_use]
pub fn terminal<C>(
    gamma: E,
    h: PublicRingKeys<'_, C>,
    records: usize,
    projection_point: &[E],
    point: &[E],
    terminal_evaluations: &[E; 9],
) -> E
where
    C: Copy + Sync + Into<i64>,
{
    let ring_weight = h.functional_evaluation(records, projection_point, point);
    let magnitude = terminal_evaluations[2]
        + E::from_i64(8) * terminal_evaluations[3]
        + E::from_i64(128) * terminal_evaluations[4]
        + E::from_i64(1_092);
    gamma * ring_weight * terminal_evaluations[5] * magnitude
}
