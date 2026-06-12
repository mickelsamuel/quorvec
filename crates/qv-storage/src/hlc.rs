//! Hybrid Logical Clock timestamp.
//!
//! The plan defines an HLC as 48-bit wall-clock milliseconds + a 16-bit counter,
//! with a node-id tiebreak, used as the per-point version for last-writer-wins.
//! The full clock (advancing on send/receive across nodes) is M4 data-plane
//! work; M2 needs the **timestamp type** and its packed wire form for the WAL
//! record. The node-id tiebreak is carried alongside as a separate field where
//! ordering needs it; the packed 64-bit value here is the (wall, counter) pair.

use serde::{Deserialize, Serialize};

/// A hybrid logical clock timestamp: 48-bit wall ms, 16-bit counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Hlc {
    /// Wall-clock milliseconds, masked to 48 bits.
    pub wall_ms: u64,
    /// Monotonic counter within a wall-ms tick.
    pub counter: u16,
}

impl Hlc {
    /// The zero timestamp (older than any real one).
    pub const ZERO: Hlc = Hlc {
        wall_ms: 0,
        counter: 0,
    };

    pub fn new(wall_ms: u64, counter: u16) -> Self {
        Self {
            wall_ms: wall_ms & 0xFFFF_FFFF_FFFF, // 48 bits
            counter,
        }
    }

    /// Pack into a single u64: 48-bit wall in the high bits, 16-bit counter low.
    /// This makes numeric comparison match logical ordering.
    pub fn pack(self) -> u64 {
        ((self.wall_ms & 0xFFFF_FFFF_FFFF) << 16) | (self.counter as u64)
    }

    /// Unpack from the 64-bit form produced by [`Hlc::pack`].
    pub fn unpack(v: u64) -> Self {
        Self {
            wall_ms: v >> 16,
            counter: (v & 0xFFFF) as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let h = Hlc::new(0x1234_5678_9ABC, 0xBEEF);
        assert_eq!(Hlc::unpack(h.pack()), h);
    }

    #[test]
    fn ordering_matches_packed() {
        let a = Hlc::new(100, 5);
        let b = Hlc::new(100, 6);
        let c = Hlc::new(101, 0);
        assert!(a < b);
        assert!(b < c);
        assert!(a.pack() < b.pack());
        assert!(b.pack() < c.pack());
    }
}
