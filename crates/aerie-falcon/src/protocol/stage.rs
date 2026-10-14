//! Shared transcript and ordinary-round mechanics for the three stages.
//!
//! Relation provers live in their named stage modules. This module deliberately
//! contains no prover registry: it only frames the common compressed round,
//! samples challenges in transcript order, and replays ordinary sumchecks.

use akita_error::AkitaError;
use jolt_field::{Field, Ring, Zero};

use super::transcript::PioTranscript;
use crate::substrate::OpeningField as E;

/// Reconstruct the omitted linear coefficient of an ordinary degree-three
/// message and evaluate it at the next challenge.
pub(crate) fn evaluate_compressed([c0, c2, c3]: [E; 3], claim: E, x: E) -> E {
    let c1 = claim - c0 - c0 - c2 - c3;
    ((c3 * x + c2) * x + c1) * x + c0
}

/// The constant polynomial for one leading inactive variable. Repeated use
/// halves the local claim at each challenge, implementing the specified
/// `2^-k` lift without expanding the witness table.
pub(crate) fn inactive_round(claim: E) -> [E; 3] {
    [
        claim * E::from_u64(2).inverse().expect("two is invertible"),
        E::zero(),
        E::zero(),
    ]
}

/// Append one compressed ordinary message and sample its challenge.
pub(crate) fn append_round(
    transcript: &mut impl PioTranscript,
    stage: &str,
    round: usize,
    message: [E; 3],
) -> Result<E, AkitaError> {
    transcript.absorb_extensions(
        &format!("aerie/falcon/v2/{stage}/round/{round}/message"),
        &message,
    )?;
    Ok(transcript.challenge_extensions(
        &format!("aerie/falcon/v2/{stage}/round/{round}/challenge"),
        1,
    )?[0])
}

/// Verify an ordinary compressed sumcheck, returning its point and terminal
/// claim for relation-specific checks.
pub(crate) fn verify_stage(
    messages: &[[E; 3]],
    rounds: usize,
    mut claim: E,
    transcript: &mut impl PioTranscript,
    stage: &str,
) -> Result<(Vec<E>, E), AkitaError> {
    if messages.len() != rounds {
        return Err(AkitaError::InvalidInput(
            "incorrect stage round count".into(),
        ));
    }
    let mut point = Vec::with_capacity(rounds);
    for (round, &message) in messages.iter().enumerate() {
        let challenge = append_round(transcript, stage, round, message)?;
        claim = evaluate_compressed(message, claim, challenge);
        point.push(challenge);
    }
    Ok((point, claim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::One;

    #[test]
    fn inactive_rounds_keep_the_local_sum_and_apply_the_terminal_scale() {
        let half = E::from_u64(2).inverse().unwrap();
        let local_sum = E::from_u64(29);
        let mut claim = local_sum;
        for challenge in [E::from_u64(7), E::from_u64(19), E::from_u64(31)] {
            let message = inactive_round(claim);
            assert_eq!(evaluate_compressed(message, claim, E::zero()), claim * half);
            assert_eq!(evaluate_compressed(message, claim, E::one()), claim * half);
            claim = evaluate_compressed(message, claim, challenge);
        }
        assert_eq!(claim, local_sum * half * half * half);
    }
}
