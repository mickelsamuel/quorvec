//! Criterion micro-benchmarks for the index layer.
//!
//! M0: brute-force insert/search baseline. M1 adds HNSW insert/search benches.
//! Numbers are recorded in the build log only — never published as claims.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use qv_hnsw::{BruteForceIndex, Metric, VectorIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn bench_bruteforce(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let mut rng = StdRng::seed_from_u64(42);

    let mut idx = BruteForceIndex::new(dim, Metric::L2);
    for id in 0..n {
        idx.insert(id as u64, &random_vec(&mut rng, dim)).unwrap();
    }

    c.bench_function("bruteforce_search_k10_n10k_d128", |b| {
        b.iter_batched(
            || random_vec(&mut rng, dim),
            |q| {
                let _ = idx.search(&q, 10, 64).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("bruteforce_insert_d128", |b| {
        b.iter_batched(
            || (rng.gen::<u64>(), random_vec(&mut rng, dim)),
            |(id, v)| {
                let mut local = BruteForceIndex::new(dim, Metric::L2);
                local.insert(id, &v).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_bruteforce);
criterion_main!(benches);
