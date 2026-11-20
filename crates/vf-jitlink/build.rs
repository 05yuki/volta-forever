fn main() {
    let map = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("version.map");
    println!("cargo:rustc-cdylib-link-arg=-Wl,--version-script={}", map.display());
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libnvJitLink.so.13");
    println!("cargo:rerun-if-changed=version.map");
}
