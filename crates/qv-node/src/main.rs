//! The quorvec node binary.
//!
//! M3: a cluster node. The metadata plane is openraft-backed; the data plane is
//! durable shards. One listen port serves both the v1 client service and the
//! internal node-to-node service.
//!
//! Run the seed node with `--bootstrap` to form a new single-node cluster; start
//! the others without it and `Join` them via the admin RPC (the integration
//! harness does this). Config via `--config node.toml` or inline flags.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use qv_node::config::NodeConfig;
use qv_node::{build_node_state, InternalService, QuorvecService};
use qv_proto::{QuorvecInternalServer, QuorvecServer};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let opts = parse_args()?;
    let cfg = opts.config;
    cfg.validate()?;
    std::fs::create_dir_all(&cfg.data_dir)?;

    let state = build_node_state(&cfg).await?;

    tracing::info!(
        node_id = cfg.node_id,
        listen = %cfg.listen_addr,
        advertise = %cfg.advertise(),
        data_dir = %cfg.data_dir.display(),
        bootstrap = opts.bootstrap,
        "quorvec node starting (M3 cluster)"
    );

    if opts.bootstrap {
        // Form a new single-node cluster with this node as the founding leader.
        state.cluster.bootstrap().await?;
        tracing::info!(node_id = cfg.node_id, "bootstrapped single-node cluster");
    }

    let v1 = QuorvecService::new(state.clone());
    let internal = InternalService::new(state.clone());

    // Hinted-handoff replay loop (M4): periodically push buffered hints to
    // replicas that have come back. Cheap when there are no hints.
    {
        let hint_state = state.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(500));
            loop {
                tick.tick().await;
                match qv_node::router::replay_hints(&hint_state).await {
                    Ok(n) if n > 0 => {
                        tracing::info!(replayed = n, "replayed hinted-handoff writes")
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(err = %e, "hint replay error"),
                }
            }
        });
    }

    // Rebalance reconcile loop (M5): each node reconciles its assigned shards
    // against the deterministic placement — acquiring shards it should hold (via
    // stream_records transfer) and dropping shards it no longer owns once the
    // hand-off is complete. Drives join rebalancing and dead-replica recovery.
    qv_node::rebalance::spawn_reconcile_loop(state.clone());

    // If this is the seed and join targets were given, admit each peer once it is
    // healthy. This lets `docker compose up` form a cluster with no extra tooling:
    // the seed bootstraps, then joins the listed nodes (id@host:port) through Raft.
    if opts.bootstrap && !opts.join_targets.is_empty() {
        let cluster = state.cluster.clone();
        let targets = opts.join_targets.clone();
        tokio::spawn(async move {
            for target in targets {
                if let Some((id, addr)) = target.split_once('@') {
                    let id: u64 = match id.parse() {
                        Ok(v) => v,
                        Err(_) => {
                            tracing::error!(%target, "bad join target (want id@host:port)");
                            continue;
                        }
                    };
                    // Wait for the peer's internal port to accept connections,
                    // then add it through Raft. Retry until it succeeds.
                    for attempt in 0..120u32 {
                        match cluster.join_node(id, addr.to_string()).await {
                            Ok(()) => {
                                tracing::info!(node_id = id, %addr, "joined peer");
                                break;
                            }
                            Err(e) => {
                                if attempt % 10 == 0 {
                                    tracing::warn!(node_id = id, %addr, err = %e, "join retry");
                                }
                                tokio::time::sleep(Duration::from_millis(500)).await;
                            }
                        }
                    }
                }
            }
        });
    }

    Server::builder()
        .add_service(QuorvecServer::new(v1))
        .add_service(QuorvecInternalServer::new(internal))
        .serve_with_shutdown(cfg.listen_addr, shutdown_signal())
        .await?;

    tracing::info!("quorvec node stopped");
    Ok(())
}

/// Parsed launch options.
struct LaunchOpts {
    config: NodeConfig,
    bootstrap: bool,
    /// Seed-only: peers to admit after bootstrap, each `id@host:port`.
    join_targets: Vec<String>,
}

/// Parse `--config <path>` or inline flags, plus `--bootstrap` and
/// `--join` (repeatable / comma-separated `id@host:port` targets).
fn parse_args() -> anyhow::Result<LaunchOpts> {
    let mut args = std::env::args().skip(1);

    let mut config_path: Option<PathBuf> = None;
    let mut node_id: Option<u64> = None;
    let mut listen: Option<SocketAddr> = None;
    // advertise is a host:port string (may be a hostname), not a SocketAddr.
    let mut advertise: Option<String> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut bootstrap = false;
    let mut join_targets: Vec<String> = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => config_path = Some(next_val(&mut args, "--config")?.into()),
            "--node-id" => node_id = Some(next_val(&mut args, "--node-id")?.parse()?),
            "--listen" => listen = Some(next_val(&mut args, "--listen")?.parse()?),
            "--advertise" => advertise = Some(next_val(&mut args, "--advertise")?),
            "--data-dir" => data_dir = Some(next_val(&mut args, "--data-dir")?.into()),
            "--bootstrap" => bootstrap = true,
            "--join" => {
                let v = next_val(&mut args, "--join")?;
                for t in v.split(',').filter(|s| !s.is_empty()) {
                    join_targets.push(t.to_string());
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other} (try --help)"),
        }
    }

    let config = if let Some(path) = config_path {
        NodeConfig::from_file(path)?
    } else {
        let listen = listen.ok_or_else(|| {
            anyhow::anyhow!("either --config or --listen (with --node-id, --data-dir) is required")
        })?;
        let toml = format!(
            "node_id = {}\nlisten_addr = \"{}\"\ndata_dir = \"{}\"\n{}",
            node_id.unwrap_or(1),
            listen,
            data_dir
                .unwrap_or_else(|| PathBuf::from("./qv-data"))
                .display()
                .to_string()
                .replace('\\', "/"),
            advertise
                .map(|a| format!("advertise_addr = \"{a}\"\n"))
                .unwrap_or_default(),
        );
        toml::from_str(&toml)?
    };

    Ok(LaunchOpts {
        config,
        bootstrap,
        join_targets,
    })
}

fn next_val(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn print_usage() {
    println!(
        "quorvec node (M3)\n\n\
         USAGE:\n  \
         qv-node --config <path.toml> [--bootstrap] [--join id@host:port,...]\n  \
         qv-node --node-id <id> --listen <addr> [--advertise <addr>] [--data-dir <dir>] [--bootstrap] [--join ...]\n\n\
         --bootstrap forms a new single-node cluster (the seed node). Peers can be\n\
         admitted by the Join admin RPC, or — on the seed — listed with --join\n\
         (id@host:port, comma-separated) so the seed admits them once healthy.\n\
         Set RUST_LOG to control log level (default: info)."
    );
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
