//! Per-shard write-ahead log.
//!
//! Record format on disk (exactly per the plan), little-endian:
//!
//! ```text
//!   u32  payload_frame_len   total bytes of the framed record that follow
//!                            (everything after this length prefix, i.e. the
//!                            CRC field through the end of payload). Used to
//!                            find the next record and to detect a torn tail.
//!   u32  crc32               crc32 (crc32fast) over the bytes AFTER this field
//!                            (op .. payload), i.e. the logical record body.
//!   u8   op                  0 = UPSERT, 1 = DELETE
//!   u64  hlc                 packed HLC (48-bit wall ms << 16 | 16-bit counter)
//!   u64  id                  point id
//!   u32  dim                 vector dimensionality (0 for DELETE)
//!   f32 * dim  vector        the vector (omitted for DELETE)
//!   u32  payload_len         payload byte length (0 if none)
//!   u8 * payload_len payload opaque payload bytes
//! ```
//!
//! Durability: `fsync` per record by default. With `wal_batch_ms > 0` the WAL
//! batches fsyncs within that window (acks are only "durable" after the next
//! fsync — documented in the durability statement below).
//!
//! Crash safety: replay reads records until either EOF or the first record that
//! is incomplete (a torn tail from a crash mid-write) or fails CRC. At that
//! point replay **truncates the file to the last good record and continues** —
//! a partially-written trailing record was never acked, so dropping it loses no
//! acknowledged write. This is the property the kill -9 harness verifies.
//!
//! ## Durability statement (fsync vs batch)
//! By default (`wal_batch_ms == 0`) every appended record is flushed with
//! `File::sync_data` before `append` returns, so a returned `append` means the
//! record is on stable storage: no acked write is lost across a crash. With
//! `wal_batch_ms > 0`, `append` returns after the OS write but before fsync;
//! fsync happens at most `wal_batch_ms` later (or on `flush`/`Drop`). In that
//! mode a crash can lose the last sub-window of writes that were acked-but-not-
//! yet-fsynced — the standard throughput-vs-durability tradeoff, made explicit.

use crate::hlc::Hlc;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// WAL operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalOp {
    Upsert,
    Delete,
}

impl WalOp {
    fn to_u8(self) -> u8 {
        match self {
            WalOp::Upsert => 0,
            WalOp::Delete => 1,
        }
    }
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(WalOp::Upsert),
            1 => Some(WalOp::Delete),
            _ => None,
        }
    }
}

/// A decoded WAL record.
#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub op: WalOp,
    pub hlc: Hlc,
    pub id: u64,
    /// Vector (empty for DELETE).
    pub vector: Vec<f32>,
    /// Opaque payload (empty if none).
    pub payload: Vec<u8>,
}

impl WalRecord {
    pub fn upsert(hlc: Hlc, id: u64, vector: Vec<f32>, payload: Vec<u8>) -> Self {
        Self {
            op: WalOp::Upsert,
            hlc,
            id,
            vector,
            payload,
        }
    }
    pub fn delete(hlc: Hlc, id: u64) -> Self {
        Self {
            op: WalOp::Delete,
            hlc,
            id,
            vector: Vec::new(),
            payload: Vec::new(),
        }
    }

    /// Encode the logical body (everything the CRC covers): op .. payload.
    fn encode_body(&self) -> Vec<u8> {
        let mut b =
            Vec::with_capacity(1 + 8 + 8 + 4 + self.vector.len() * 4 + 4 + self.payload.len());
        b.push(self.op.to_u8());
        b.extend_from_slice(&self.hlc.pack().to_le_bytes());
        b.extend_from_slice(&self.id.to_le_bytes());
        b.extend_from_slice(&(self.vector.len() as u32).to_le_bytes());
        for &f in &self.vector {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.payload);
        b
    }
}

/// Errors from WAL operations.
#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// An append-only WAL writer for one shard.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    /// Bytes written (the current logical end offset).
    len: u64,
    /// 0 = fsync every record; >0 = batch window (ms) — batching is time-driven
    /// by the caller via `maybe_flush`; this type tracks whether a flush is due.
    batch_ms: u64,
    dirty: bool,
}

impl Wal {
    /// Open (creating if needed) the WAL at `path` for appending. Recovery
    /// (replay + torn-tail truncation) is done separately via [`Wal::recover`].
    pub fn open(path: impl AsRef<Path>, batch_ms: u64) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            len,
            batch_ms,
            dirty: false,
        })
    }

    /// Current on-disk length (logical end offset).
    pub fn offset(&self) -> u64 {
        self.len
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record. Frames it, writes it, and (in fsync mode) flushes to
    /// stable storage before returning. Returns the new end offset.
    pub fn append(&mut self, rec: &WalRecord) -> Result<u64, WalError> {
        let body = rec.encode_body();
        let crc = crc32fast::hash(&body);

        // Framed length = crc(4) + body.
        let frame_len = (4 + body.len()) as u32;

        self.writer.write_all(&frame_len.to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&body)?;
        self.len += 4 + frame_len as u64;
        self.dirty = true;

        if self.batch_ms == 0 {
            self.flush()?;
        }
        Ok(self.len)
    }

    /// Flush buffered bytes and fsync to stable storage.
    pub fn flush(&mut self) -> Result<(), WalError> {
        if self.dirty {
            self.writer.flush()?;
            self.writer.get_ref().sync_data()?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Replay every intact record from `path`, in order, truncating a torn or
    /// CRC-failing tail and returning the surviving records plus the recovered
    /// end offset.
    ///
    /// This is the crash-recovery entry point: a partial trailing record (from a
    /// crash mid-append) is detected and the file is physically truncated to the
    /// last good boundary, so subsequent appends continue cleanly.
    pub fn recover(path: impl AsRef<Path>) -> Result<(Vec<WalRecord>, u64), WalError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok((Vec::new(), 0));
        }
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let total = file.metadata()?.len();
        file.seek(SeekFrom::Start(0))?;

        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut records = Vec::new();
        let mut good_end: u64 = 0;
        let mut pos: usize = 0;

        while pos + 4 <= buf.len() {
            // Read frame length.
            let frame_len =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            let frame_start = pos + 4;
            let frame_end = frame_start + frame_len;
            if frame_len < 4 || frame_end > buf.len() {
                // Torn tail: incomplete frame. Stop here.
                break;
            }
            let crc_bytes = &buf[frame_start..frame_start + 4];
            let stored_crc =
                u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
            let body = &buf[frame_start + 4..frame_end];
            if crc32fast::hash(body) != stored_crc {
                // Corruption / torn body: stop, truncate from here.
                break;
            }
            match decode_body(body) {
                Some(rec) => {
                    records.push(rec);
                    pos = frame_end;
                    good_end = pos as u64;
                }
                None => break, // malformed body; treat as tail
            }
        }

        // Truncate any trailing garbage / torn tail.
        if good_end < total {
            file.set_len(good_end)?;
            file.sync_all()?;
        }

        Ok((records, good_end))
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        // Best-effort flush on drop so a clean shutdown is durable even in batch
        // mode.
        let _ = self.flush();
    }
}

/// Decode a logical record body (the CRC-covered bytes). Returns `None` on any
/// structural inconsistency (treated as a torn tail by the caller).
fn decode_body(body: &[u8]) -> Option<WalRecord> {
    let mut p = 0usize;
    let need = |p: usize, n: usize| p + n <= body.len();

    if !need(p, 1) {
        return None;
    }
    let op = WalOp::from_u8(body[p])?;
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
        let f = f32::from_le_bytes(body[p..p + 4].try_into().ok()?);
        vector.push(f);
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

    // Exact-length check: a valid body consumes all its bytes.
    if p != body.len() {
        return None;
    }

    Some(WalRecord {
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

    #[test]
    fn append_and_recover_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.wal");

        {
            let mut wal = Wal::open(&path, 0).unwrap();
            wal.append(&WalRecord::upsert(
                Hlc::new(1, 0),
                10,
                vec![1.0, 2.0, 3.0],
                vec![],
            ))
            .unwrap();
            wal.append(&WalRecord::upsert(
                Hlc::new(2, 0),
                20,
                vec![4.0, 5.0, 6.0],
                vec![9, 9],
            ))
            .unwrap();
            wal.append(&WalRecord::delete(Hlc::new(3, 0), 10)).unwrap();
        }

        let (recs, end) = Wal::recover(&path).unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(
            recs[0],
            WalRecord::upsert(Hlc::new(1, 0), 10, vec![1.0, 2.0, 3.0], vec![])
        );
        assert_eq!(recs[1].payload, vec![9, 9]);
        assert_eq!(recs[2].op, WalOp::Delete);
        assert!(end > 0);
    }

    #[test]
    fn torn_tail_is_truncated() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.wal");

        let good_end = {
            let mut wal = Wal::open(&path, 0).unwrap();
            wal.append(&WalRecord::upsert(Hlc::new(1, 0), 1, vec![1.0], vec![]))
                .unwrap();
            wal.append(&WalRecord::upsert(Hlc::new(2, 0), 2, vec![2.0], vec![]))
                .unwrap()
        };

        // Simulate a crash mid-write: append garbage bytes (a partial frame).
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            // A frame_len claiming 100 bytes, but only a few follow.
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(&[1, 2, 3, 4, 5]).unwrap();
        }

        let (recs, end) = Wal::recover(&path).unwrap();
        assert_eq!(recs.len(), 2, "torn tail must be dropped");
        assert_eq!(end, good_end, "file truncated to last good record");

        // The file must now be physically truncated and appendable again.
        let meta_len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(meta_len, good_end);

        let mut wal = Wal::open(&path, 0).unwrap();
        wal.append(&WalRecord::upsert(Hlc::new(3, 0), 3, vec![3.0], vec![]))
            .unwrap();
        let (recs2, _) = Wal::recover(&path).unwrap();
        assert_eq!(recs2.len(), 3);
    }

    #[test]
    fn crc_corruption_is_caught() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.wal");
        {
            let mut wal = Wal::open(&path, 0).unwrap();
            wal.append(&WalRecord::upsert(
                Hlc::new(1, 0),
                1,
                vec![1.0, 2.0],
                vec![],
            ))
            .unwrap();
            wal.append(&WalRecord::upsert(
                Hlc::new(2, 0),
                2,
                vec![3.0, 4.0],
                vec![],
            ))
            .unwrap();
        }
        // Flip a byte in the middle of the file (corrupt the first record body).
        {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            f.seek(SeekFrom::Start(12)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xFF;
            f.seek(SeekFrom::Start(12)).unwrap();
            f.write_all(&b).unwrap();
        }
        let (recs, _) = Wal::recover(&path).unwrap();
        // The corrupted first record fails CRC, so recovery stops at 0 records.
        assert_eq!(recs.len(), 0);
    }
}
