fn main() {
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libvf_audit.so");
}
