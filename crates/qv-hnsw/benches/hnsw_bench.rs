//! Criterion micro-benchmarks for the index layer.
//!
//! Brute-force baseline (M0) plus HNSW insert/search (M1). Numbers are recorded
//! in the build log only — never published as claims.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use qv_hnsw::{BruteForceIndex, HnswIndex, Metric, VectorIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 128;

fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn bench_bruteforce(c: &mut Criterion) {
    let n = 10_000;
    let mut rng = StdRng::seed_from_u64(42);

    let mut idx = BruteForceIndex::new(DIM, Metric::L2);
    for id in 0..n {
        idx.insert(id as u64, &random_vec(&mut rng, DIM)).unwrap();
    }

    c.bench_function("bruteforce_search_k10_n10k_d128", |b| {
        b.iter_batched(
            || random_vec(&mut rng, DIM),
            |q| {
                let _ = idx.search(&q, 10, 64).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_hnsw(c: &mut Criterion) {
    let n = 10_000;
    let mut rng = StdRng::seed_from_u64(7);

    // Pre-build a 10k index for search benches.
    let mut idx = HnswIndex::new(DIM, Metric::L2);
    for id in 0..n {
        idx.insert(id as u64, &random_vec(&mut rng, DIM)).unwrap();
    }

    c.bench_function("hnsw_search_k10_ef64_n10k_d128", |b| {
        b.iter_batched(
            || random_vec(&mut rng, DIM),
            |q| {
                let _ = idx.search(&q, 10, 64).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    // Insert bench: amortized single insert into a warm 10k index.
    c.bench_function("hnsw_insert_into_10k_d128", |b| {
        let mut next_id = n as u64;
        b.iter_batched(
            || random_vec(&mut rng, DIM),
            |v| {
                idx.insert(next_id, &v).unwrap();
                next_id += 1;
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_bruteforce, bench_hnsw);
criterion_main!(benches);
