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

/// A monotonic hybrid-logical-clock generator.
///
/// `now()` returns a strictly increasing [`Hlc`]: it tracks the max of wall-clock
/// milliseconds and the last issued timestamp, bumping the 16-bit counter when
/// the wall clock has not advanced since the last call (and carrying into the
/// wall component if the counter saturates). M4's coordinator advances this on
/// receiving a peer's timestamp (`update`), giving the full HLC; M3 uses `now()`
/// for a locally-monotonic version stamp on the single-replica write path.
#[derive(Debug)]
pub struct HlcClock {
    last: std::sync::Mutex<Hlc>,
}

impl Default for HlcClock {
    fn default() -> Self {
        Self::new()
    }
}

impl HlcClock {
    pub fn new() -> Self {
        Self {
            last: std::sync::Mutex::new(Hlc::ZERO),
        }
    }

    fn wall_now_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
            & 0xFFFF_FFFF_FFFF
    }

    /// Issue the next strictly-increasing timestamp.
    pub fn now(&self) -> Hlc {
        let mut last = self.last.lock().unwrap();
        let wall = Self::wall_now_ms();
        let next = if wall > last.wall_ms {
            Hlc::new(wall, 0)
        } else {
            // Wall clock did not advance: bump the counter (carry into wall if it
            // saturates so the result still strictly increases).
            match last.counter.checked_add(1) {
                Some(c) => Hlc::new(last.wall_ms, c),
                None => Hlc::new(last.wall_ms + 1, 0),
            }
        };
        *last = next;
        next
    }

    /// Advance the clock on receiving a remote timestamp, then issue a new local
    /// timestamp strictly greater than both local and remote (the HLC receive
    /// rule). Used by M4's replica write path.
    pub fn update(&self, remote: Hlc) -> Hlc {
        let mut last = self.last.lock().unwrap();
        let wall = Self::wall_now_ms();
        let max_wall = wall.max(last.wall_ms).max(remote.wall_ms);
        let next = if max_wall == last.wall_ms && max_wall == remote.wall_ms {
            Hlc::new(max_wall, last.counter.max(remote.counter).saturating_add(1))
        } else if max_wall == last.wall_ms {
            Hlc::new(max_wall, last.counter.saturating_add(1))
        } else if max_wall == remote.wall_ms {
            Hlc::new(max_wall, remote.counter.saturating_add(1))
        } else {
            Hlc::new(max_wall, 0)
        };
        *last = next;
        next
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

    #[test]
    fn clock_is_strictly_monotonic() {
        let clock = HlcClock::new();
        let mut prev = Hlc::ZERO;
        for _ in 0..10_000 {
            let t = clock.now();
            assert!(t > prev, "HLC not strictly increasing: {t:?} !> {prev:?}");
            prev = t;
        }
    }

    #[test]
    fn update_dominates_remote() {
        let clock = HlcClock::new();
        let local = clock.now();
        // A remote timestamp far in the future.
        let remote = Hlc::new(local.wall_ms + 1000, 50);
        let after = clock.update(remote);
        assert!(after > remote, "update must exceed remote");
        assert!(after > local, "update must exceed prior local");
    }
}
