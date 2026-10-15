//! Nine authenticated evaluations at the four groups' three native points.
use super::{manifest::GroupRole, stage3::Stage3Output, FalconDegree};
use crate::substrate::OpeningField as E;
#[derive(Clone, Debug, PartialEq)]
pub struct GroupOpening {
    pub role: GroupRole,
    pub point: Vec<E>,
    pub values: Vec<E>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct OpeningLedger {
    pub groups: Vec<GroupOpening>,
}
#[must_use]
pub fn opening_ledger(degree: FalconDegree, output: &Stage3Output) -> OpeningLedger {
    let point = &output.point;
    let v = &output.terminal_evaluations;
    OpeningLedger {
        groups: vec![
            GroupOpening {
                role: GroupRole::S1,
                point: point.clone(),
                values: vec![v[0]],
            },
            GroupOpening {
                role: GroupRole::Slack,
                point: point[degree.logn() - 2..].to_vec(),
                values: vec![v[1]],
            },
            GroupOpening {
                role: GroupRole::S2Sources,
                point: point.clone(),
                values: v[2..6].to_vec(),
            },
            GroupOpening {
                role: GroupRole::Budget,
                point: point[degree.logn()..].to_vec(),
                values: v[6..].to_vec(),
            },
        ],
    }
}
