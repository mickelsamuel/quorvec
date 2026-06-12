//! Distance kernels with a SIMD path (simsimd) and a scalar reference fallback.
//!
//! The scalar implementations are the source of truth for correctness; the SIMD
//! path is checked against them within an fp tolerance in unit tests. Both
//! return values where *smaller means closer*:
//!   - [`Metric::L2`] returns squared euclidean distance.
//!   - [`Metric::Cosine`] returns cosine distance `1 - cos_sim`.

use serde::{Deserialize, Serialize};

/// Distance metric for a collection of vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Squared euclidean distance (no sqrt — monotonic, cheaper, exact ordering).
    L2,
    /// Cosine distance, `1 - cosine_similarity`.
    Cosine,
}

impl Metric {
    /// Distance between two equal-length vectors under this metric.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`. Callers (the index layer) guarantee
    /// dimension agreement before this is reached.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "distance() on mismatched dims");
        match self {
            Metric::L2 => l2_sq(a, b),
            Metric::Cosine => cosine_distance(a, b),
        }
    }
}

// ---- Public kernel entry points (dispatch SIMD vs scalar) -------------------

/// Squared L2 distance.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        // simsimd returns the squared euclidean distance as f64.
        simsimd::SpatialSimilarity::sqeuclidean(a, b)
            .map(|d| d as f32)
            .unwrap_or_else(|| l2_sq_scalar(a, b))
    }
    #[cfg(not(feature = "simd"))]
    {
        l2_sq_scalar(a, b)
    }
}

/// Cosine distance, `1 - cosine_similarity`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        // simsimd::cosine already returns a *distance* (1 - similarity).
        simsimd::SpatialSimilarity::cosine(a, b)
            .map(|d| d as f32)
            .unwrap_or_else(|| cosine_distance_scalar(a, b))
    }
    #[cfg(not(feature = "simd"))]
    {
        cosine_distance_scalar(a, b)
    }
}

// ---- Scalar reference implementations (correctness source of truth) --------

/// Scalar squared L2. Uses a running sum; adequate fp behavior for the
/// dimensionalities quorvec targets (<= a few thousand).
pub fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Scalar cosine distance `1 - (a·b)/(|a||b|)`. Zero-norm vectors yield a
/// distance of 1.0 (maximally far) by convention, avoiding NaN.
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt()) * (nb.sqrt());
    if denom == 0.0 {
        return 1.0;
    }
    let sim = dot / denom;
    // Clamp to guard against fp drift just outside [-1, 1].
    let sim = sim.clamp(-1.0, 1.0);
    1.0 - sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_known_values() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 2.0];
        // squared distance = 1 + 4 + 4 = 9
        assert!((l2_sq_scalar(&a, &b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_known_values() {
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 0.0];
        assert!(cosine_distance_scalar(&a, &b).abs() < 1e-6); // identical -> 0
        let c = [0.0f32, 1.0];
        assert!((cosine_distance_scalar(&a, &c) - 1.0).abs() < 1e-6); // orthogonal -> 1
        let d = [-1.0f32, 0.0];
        assert!((cosine_distance_scalar(&a, &d) - 2.0).abs() < 1e-6); // opposite -> 2
    }

    #[test]
    fn cosine_zero_norm_is_one() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        assert!((cosine_distance_scalar(&a, &b) - 1.0).abs() < 1e-6);
    }

    // When the simd feature is on, the SIMD path must agree with scalar.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_matches_scalar() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.013 - 0.7).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32) * -0.021 + 0.3).collect();

        let l2_s = l2_sq_scalar(&a, &b);
        let l2_v = l2_sq(&a, &b);
        assert!(
            (l2_s - l2_v).abs() <= 1e-3 * l2_s.max(1.0),
            "L2 simd {l2_v} vs scalar {l2_s}"
        );

        let cos_s = cosine_distance_scalar(&a, &b);
        let cos_v = cosine_distance(&a, &b);
        assert!(
            (cos_s - cos_v).abs() <= 1e-4,
            "cosine simd {cos_v} vs scalar {cos_s}"
        );
    }
}
