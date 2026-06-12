//! The quorvec node binary.
//!
//! M0: single-node gRPC server over an in-memory brute-force index. Run with a
//! TOML config (`qv-node --config node.toml`) or with inline flags
//! (`qv-node --node-id 1 --listen 127.0.0.1:7000 --data-dir ./data`).

use qv_node::config::NodeConfig;
use qv_node::service::QuorvecService;
use qv_node::store::Store;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;

use qv_proto::QuorvecServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = parse_args()?;
    cfg.validate()?;

    std::fs::create_dir_all(&cfg.data_dir)?;

    let store = Arc::new(Store::new());
    let svc = QuorvecService::new(store, cfg.node_id, cfg.advertise().to_string());

    tracing::info!(
        node_id = cfg.node_id,
        listen = %cfg.listen_addr,
        data_dir = %cfg.data_dir.display(),
        "quorvec node starting (M0 single-node)"
    );

    Server::builder()
        .add_service(QuorvecServer::new(svc))
        .serve_with_shutdown(cfg.listen_addr, shutdown_signal())
        .await?;

    tracing::info!("quorvec node stopped");
    Ok(())
}

/// Parse either `--config <path>` or inline flags into a [`NodeConfig`].
fn parse_args() -> anyhow::Result<NodeConfig> {
    let mut args = std::env::args().skip(1);

    let mut config_path: Option<PathBuf> = None;
    let mut node_id: Option<u64> = None;
    let mut listen: Option<SocketAddr> = None;
    let mut advertise: Option<SocketAddr> = None;
    let mut data_dir: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => config_path = Some(next_val(&mut args, "--config")?.into()),
            "--node-id" => node_id = Some(next_val(&mut args, "--node-id")?.parse()?),
            "--listen" => listen = Some(next_val(&mut args, "--listen")?.parse()?),
            "--advertise" => advertise = Some(next_val(&mut args, "--advertise")?.parse()?),
            "--data-dir" => data_dir = Some(next_val(&mut args, "--data-dir")?.into()),
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other} (try --help)"),
        }
    }

    if let Some(path) = config_path {
        return Ok(NodeConfig::from_file(path)?);
    }

    // Inline-flag path: build a config with defaults for the cluster fields.
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
    Ok(toml::from_str(&toml)?)
}

fn next_val(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn print_usage() {
    println!(
        "quorvec node (M0)\n\n\
         USAGE:\n  \
         qv-node --config <path.toml>\n  \
         qv-node --node-id <id> --listen <addr> [--advertise <addr>] [--data-dir <dir>]\n\n\
         Set RUST_LOG to control log level (default: info)."
    );
}

/// Resolve on Ctrl-C so the server shuts down cleanly.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
