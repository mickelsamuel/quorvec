//! Node configuration (TOML) with fail-fast validation.
//!
//! The schema matches the plan's config block. Cluster-only fields (raft peers,
//! vnodes) are parsed and validated now but unused until M3; they live here so
//! the config surface is stable from the start.

use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Top-level node config.
//
// Several fields (raft, wal_batch_ms, vnodes) are parsed and validated now but
// only consumed from M2/M3 onward; allow dead_code so the config surface can be
// stable from the start without tripping `clippy -D warnings`.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct NodeConfig {
    /// Stable numeric id for this node.
    pub node_id: u64,
    /// Address the gRPC server binds to.
    pub listen_addr: SocketAddr,
    /// Address other nodes use to reach this one (defaults to listen_addr).
    #[serde(default)]
    pub advertise_addr: Option<SocketAddr>,
    /// Directory for shard data (WAL, snapshots). Created if missing.
    pub data_dir: PathBuf,
    /// Raft seed peers for the metadata plane (M3+; ignored single-node).
    #[serde(default)]
    pub raft: RaftConfig,
    /// WAL fsync batching window in ms; 0 = fsync every record (M2).
    #[serde(default)]
    pub wal_batch_ms: u64,
    /// Snapshot trigger threshold in MB of WAL (M2).
    #[serde(default = "default_snapshot_wal_mb")]
    pub snapshot_wal_mb: u64,
    /// Virtual nodes per physical node on the hash ring (M3+).
    #[serde(default = "default_vnodes")]
    pub vnodes: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)] // peers consumed by the metadata plane from M3.
pub struct RaftConfig {
    /// Seed list of peer advertise addresses.
    #[serde(default)]
    pub peers: Vec<String>,
}

fn default_snapshot_wal_mb() -> u64 {
    128
}

fn default_vnodes() -> u32 {
    64
}

/// Config validation errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
}

impl NodeConfig {
    /// Load and validate a config from a TOML file.
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self, ConfigError> {
        let path = path.into();
        let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
            path: path.clone(),
            source,
        })?;
        let cfg: NodeConfig = toml::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Fail-fast invariant checks.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vnodes == 0 {
            return Err(ConfigError::Invalid("vnodes must be > 0".into()));
        }
        if self.snapshot_wal_mb == 0 {
            return Err(ConfigError::Invalid("snapshot_wal_mb must be > 0".into()));
        }
        Ok(())
    }

    /// The advertise address, falling back to the listen address.
    pub fn advertise(&self) -> SocketAddr {
        self.advertise_addr.unwrap_or(self.listen_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let toml = r#"
            node_id = 1
            listen_addr = "127.0.0.1:7000"
            data_dir = "/tmp/qv"
        "#;
        let cfg: NodeConfig = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.node_id, 1);
        assert_eq!(cfg.vnodes, 64);
        assert_eq!(cfg.snapshot_wal_mb, 128);
        assert_eq!(cfg.wal_batch_ms, 0);
        assert_eq!(cfg.advertise(), cfg.listen_addr);
    }

    #[test]
    fn rejects_zero_vnodes() {
        let toml = r#"
            node_id = 1
            listen_addr = "127.0.0.1:7000"
            data_dir = "/tmp/qv"
            vnodes = 0
        "#;
        let cfg: NodeConfig = toml::from_str(toml).unwrap();
        assert!(cfg.validate().is_err());
    }
}
