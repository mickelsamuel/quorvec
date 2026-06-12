//! M1 property tests (proptest): random sequences of insert/delete must keep
//! the HNSW graph's structural invariants, and search must never return a
//! tombstoned id.

use proptest::prelude::*;
use qv_hnsw::{HnswIndex, Metric, VectorIndex};

const DIM: usize = 8;

/// An operation in a random program.
#[derive(Debug, Clone)]
enum Op {
    Insert(u64, Vec<f32>),
    Delete(u64),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let vec = prop::collection::vec(-10.0f32..10.0, DIM..=DIM);
    prop_oneof![
        // Bias toward inserts so the graph grows.
        3 => (0u64..64, vec.clone()).prop_map(|(id, v)| Op::Insert(id, v)),
        1 => (0u64..64).prop_map(Op::Delete),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn invariants_hold_under_random_ops(ops in prop::collection::vec(op_strategy(), 1..200)) {
        let mut idx = HnswIndex::new(DIM, Metric::L2);
        // Mirror of which ids are currently live, to validate search filtering.
        let mut live: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for op in &ops {
            match op {
                Op::Insert(id, v) => {
                    idx.insert(*id, v).unwrap();
                    live.insert(*id);
                }
                Op::Delete(id) => {
                    let removed = idx.delete(*id);
                    if removed {
                        live.remove(id);
                    }
                }
            }
            // Invariant 1-6: structure stays consistent after every op.
            idx.validate_invariants().map_err(TestCaseError::fail)?;
        }

        // len() matches the live set.
        prop_assert_eq!(idx.len(), live.len());

        // Search never returns a tombstoned id, and never more than k.
        let query = vec![0.0f32; DIM];
        let res = idx.search(&query, 10, 64).unwrap();
        prop_assert!(res.len() <= 10);
        for (id, _score) in &res {
            prop_assert!(live.contains(id), "search returned non-live id {}", id);
        }

        // get_vector agrees with liveness.
        for id in 0u64..64 {
            let got = idx.get_vector(id);
            prop_assert_eq!(got.is_some(), live.contains(&id));
        }
    }
}
