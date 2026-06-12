use std::path::PathBuf;

// Compile proto/quorvec.proto into prost messages + tonic service stubs.
//
// We point the PROTOC env var at the precompiled binary shipped by
// protoc-bin-vendored so that neither developer machines nor CI need a
// system protoc on PATH. PROTOC_NO_VENDOR is intentionally NOT set, but the
// explicit override below takes precedence regardless.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("protoc-bin-vendored should provide a protoc binary for this host");
    // SAFETY: single-threaded build script, set before any codegen runs.
    unsafe {
        std::env::set_var("PROTOC", &protoc);
    }

    let proto_root = workspace_proto_dir();
    let proto_file = proto_root.join("quorvec.proto");

    println!("cargo:rerun-if-changed={}", proto_file.display());

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[proto_file], &[proto_root])?;

    Ok(())
}

// proto/ lives at the workspace root, two levels up from crates/qv-proto.
fn workspace_proto_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .map(|p| p.join("proto"))
        .expect("qv-proto manifest should be at <root>/crates/qv-proto")
}
