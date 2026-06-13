//! WAL-persisted hinted handoff (M4).
//!
//! When a coordinator cannot reach a replica for a write, it records a **hint**:
//! the point operation plus the target replica it was meant for. The hint is
//! written to a durable, append-only log on the *hint-holder* node (the next
//! distinct ring node). When the target replica comes back, the holder replays
//! the buffered hints to it and truncates the log.
//!
//! The format mirrors the shard WAL exactly (length-prefixed, CRC32-checked,
//! torn-tail-safe) so the same crash-safety reasoning applies: a half-written
//! hint at the tail is dropped on recovery, and it was never acked as buffered.
//! Each hint additionally carries the **target node id** and the **collection /
//! shard** it belongs to, so the holder knows where to replay it.
//!
//! Hints are grouped on disk in one file per holder; replay filters by target
//! node id. This is the simple-correct v1 design; per-target files are a labeled
//! future refinement.

use crate::hlc::Hlc;
use crate::wal::WalOp;
use std::fs::OpenOptions;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// One buffered write destined for a temporarily-unreachable replica.
#[derive(Debug, Clone, PartialEq)]
pub struct Hint {
    /// The replica node this write was meant for.
    pub target_node: u64,
    pub collection: String,
    pub shard_idx: u32,
    pub op: WalOp,
    pub hlc: Hlc,
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum HintError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// A durable, append-only hint log for one holder node.
pub struct HintLog {
    path: PathBuf,
    writer: BufWriter<std::fs::File>,
    dirty: bool,
}

impl HintLog {
    /// Open (creating if needed) the hint log at `path` for appending.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HintError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            dirty: false,
        })
    }

    /// Append a hint and fsync it (a buffered hint must survive a crash, or the
    /// write it stands in for is silently lost).
    pub fn append(&mut self, hint: &Hint) -> Result<(), HintError> {
        let body = encode_hint(hint);
        let crc = crc32fast::hash(&body);
        let frame_len = (4 + body.len()) as u32;
        self.writer.write_all(&frame_len.to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&body)?;
        self.dirty = true;
        self.flush()
    }

    /// Flush + fsync buffered bytes.
    pub fn flush(&mut self) -> Result<(), HintError> {
        if self.dirty {
            self.writer.flush()?;
            self.writer.get_ref().sync_data()?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Read every intact hint (torn/CRC-failing tail dropped, like the WAL).
    pub fn read_all(path: impl AsRef<Path>) -> Result<Vec<Hint>, HintError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut file = OpenOptions::new().read(true).open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut hints = Vec::new();
        let mut pos = 0usize;
        while pos + 4 <= buf.len() {
            let frame_len =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            let start = pos + 4;
            let end = start + frame_len;
            if frame_len < 4 || end > buf.len() {
                break; // torn tail
            }
            let stored_crc =
                u32::from_le_bytes([buf[start], buf[start + 1], buf[start + 2], buf[start + 3]]);
            let body = &buf[start + 4..end];
            if crc32fast::hash(body) != stored_crc {
                break;
            }
            match decode_hint(body) {
                Some(h) => {
                    hints.push(h);
                    pos = end;
                }
                None => break,
            }
        }
        Ok(hints)
    }

    /// Remove the hint file entirely (after a successful full replay). A new one
    /// is created lazily on the next [`HintLog::open`].
    pub fn clear(path: impl AsRef<Path>) -> Result<(), HintError> {
        let path = path.as_ref();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn encode_hint(h: &Hint) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&h.target_node.to_le_bytes());
    b.extend_from_slice(&(h.collection.len() as u32).to_le_bytes());
    b.extend_from_slice(h.collection.as_bytes());
    b.extend_from_slice(&h.shard_idx.to_le_bytes());
    b.push(match h.op {
        WalOp::Upsert => 0,
        WalOp::Delete => 1,
    });
    b.extend_from_slice(&h.hlc.pack().to_le_bytes());
    b.extend_from_slice(&h.id.to_le_bytes());
    b.extend_from_slice(&(h.vector.len() as u32).to_le_bytes());
    for &f in &h.vector {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b.extend_from_slice(&(h.payload.len() as u32).to_le_bytes());
    b.extend_from_slice(&h.payload);
    b
}

fn decode_hint(body: &[u8]) -> Option<Hint> {
    let mut p = 0usize;
    let need = |p: usize, n: usize| p + n <= body.len();

    if !need(p, 8) {
        return None;
    }
    let target_node = u64::from_le_bytes(body[p..p + 8].try_into().ok()?);
    p += 8;

    if !need(p, 4) {
        return None;
    }
    let clen = u32::from_le_bytes(body[p..p + 4].try_into().ok()?) as usize;
    p += 4;
    if !need(p, clen) {
        return None;
    }
    let collection = String::from_utf8(body[p..p + clen].to_vec()).ok()?;
    p += clen;

    if !need(p, 4) {
        return None;
    }
    let shard_idx = u32::from_le_bytes(body[p..p + 4].try_into().ok()?);
    p += 4;

    if !need(p, 1) {
        return None;
    }
    let op = match body[p] {
        0 => WalOp::Upsert,
        1 => WalOp::Delete,
        _ => return None,
    };
    p += 1;

    if !need(p, 8) {
        return None;
    }
    let hlc = Hlc::unpack(u64::from_le_bytes(body[p..p + 8].try_into().ok()?));
    p += 8;

    if !need(p, 8) {
        return None;
    }
    let id = u64::from_le_bytes(body[p..p + 8].try_into().ok()?);
    p += 8;

    if !need(p, 4) {
        return None;
    }
    let dim = u32::from_le_bytes(body[p..p + 4].try_into().ok()?) as usize;
    p += 4;
    if !need(p, dim * 4) {
        return None;
    }
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        vector.push(f32::from_le_bytes(body[p..p + 4].try_into().ok()?));
        p += 4;
    }

    if !need(p, 4) {
        return None;
    }
    let plen = u32::from_le_bytes(body[p..p + 4].try_into().ok()?) as usize;
    p += 4;
    if !need(p, plen) {
        return None;
    }
    let payload = body[p..p + plen].to_vec();
    p += plen;

    if p != body.len() {
        return None;
    }

    Some(Hint {
        target_node,
        collection,
        shard_idx,
        op,
        hlc,
        id,
        vector,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn hint(target: u64, id: u64, hlc: u64, del: bool) -> Hint {
        Hint {
            target_node: target,
            collection: "c".into(),
            shard_idx: 2,
            op: if del { WalOp::Delete } else { WalOp::Upsert },
            hlc: Hlc::new(hlc, 0),
            id,
            vector: if del { vec![] } else { vec![1.0, 2.0] },
            payload: if del { vec![] } else { vec![7] },
        }
    }

    #[test]
    fn append_and_read_roundtrip() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("hints.log");
        {
            let mut log = HintLog::open(&p).unwrap();
            log.append(&hint(3, 10, 1, false)).unwrap();
            log.append(&hint(3, 11, 2, false)).unwrap();
            log.append(&hint(4, 10, 3, true)).unwrap();
        }
        let all = HintLog::read_all(&p).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], hint(3, 10, 1, false));
        assert_eq!(all[2].op, WalOp::Delete);
        // Filtering by target node is the holder's replay strategy.
        let for_3: Vec<_> = all.iter().filter(|h| h.target_node == 3).collect();
        assert_eq!(for_3.len(), 2);
    }

    #[test]
    fn torn_tail_is_dropped() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("hints.log");
        {
            let mut log = HintLog::open(&p).unwrap();
            log.append(&hint(3, 10, 1, false)).unwrap();
        }
        // Append a partial frame (simulated crash mid-write).
        {
            let mut f = OpenOptions::new().append(true).open(&p).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&[1, 2, 3]).unwrap();
        }
        let all = HintLog::read_all(&p).unwrap();
        assert_eq!(all.len(), 1, "torn tail must be dropped");
    }

    #[test]
    fn clear_removes_file() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("hints.log");
        {
            let mut log = HintLog::open(&p).unwrap();
            log.append(&hint(3, 10, 1, false)).unwrap();
        }
        HintLog::clear(&p).unwrap();
        assert!(HintLog::read_all(&p).unwrap().is_empty());
    }
}
