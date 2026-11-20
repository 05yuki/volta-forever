//! Generates the pass-through half of the library from the export list of NVIDIA's `libcuda.so`:
//! jumps for every function that `src/hooks.rs` does not define, the name table, and the lists of
//! this library's own entry points.

use std::fmt::Write;
use std::{env, fs, path::Path};

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let symbols = fs::read_to_string(format!("{dir}/symbols/libcuda-580.178.04.txt")).unwrap();
    let names: Vec<&str> = symbols.split_whitespace().collect();
    let hooks = fs::read_to_string(format!("{dir}/src/hooks.rs")).unwrap();
    let hooked = |name: &str| hooks.contains(&format!("extern \"C\" fn {name}("));

    let mut out = String::new();
    writeln!(out, "pub const NSYM: usize = {};", names.len()).unwrap();
    writeln!(out, "pub static NAMES: [&core::ffi::CStr; NSYM] = [").unwrap();
    for name in &names {
        writeln!(out, "    c\"{name}\",").unwrap();
    }
    writeln!(out, "];\n#[allow(non_upper_case_globals, dead_code)]\npub mod idx {{").unwrap();
    for (i, name) in names.iter().enumerate() {
        writeln!(out, "    pub const {name}: usize = {i};").unwrap();
    }
    writeln!(out, "}}").unwrap();
    // Two jumps per forwarded function: the exported one, and one that also counts the call, handed out
    // to the runtime only when statistics are asked for.
    for (i, name) in names.iter().enumerate() {
        if hooked(name) {
            continue;
        }
        let offset = 8 * i;
        writeln!(
            out,
            "#[unsafe(naked)]\n#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn {name}() {{\n    \
             core::arch::naked_asm!(\"jmp qword ptr [rip + {{reals}} + {offset}]\", reals = sym crate::REALS)\n}}"
        )
        .unwrap();
        writeln!(
            out,
            "#[unsafe(naked)]\nunsafe extern \"C\" fn counted_{name}() {{\n    \
             core::arch::naked_asm!(\"lock inc qword ptr [rip + {{counts}} + {offset}]\", \
             \"jmp qword ptr [rip + {{reals}} + {offset}]\", counts = sym crate::COUNTS, reals = sym crate::REALS)\n}}"
        )
        .unwrap();
    }
    writeln!(
        out,
        "/// This library's entry point for each function: the hook, or the counting jump."
    )
    .unwrap();
    writeln!(out, "pub fn counted_entries() -> [usize; NSYM] {{\n    [").unwrap();
    for name in &names {
        let path = if hooked(name) { "crate::hooks::" } else { "counted_" };
        writeln!(out, "        {path}{name} as *const () as usize,").unwrap();
    }
    writeln!(out, "    ]\n}}").unwrap();
    writeln!(out, "/// This library's exported entry point for each function.").unwrap();
    writeln!(out, "pub fn own_entries() -> [usize; NSYM] {{\n    [").unwrap();
    for name in &names {
        let path = if hooked(name) { "crate::hooks::" } else { "" };
        writeln!(out, "        {path}{name} as *const () as usize,").unwrap();
    }
    writeln!(out, "    ]\n}}").unwrap();
    writeln!(out, "pub static HOOKED: [bool; NSYM] = [").unwrap();
    for name in &names {
        writeln!(out, "    {},", hooked(name)).unwrap();
    }
    writeln!(out, "];").unwrap();

    // The identity of this build, which is in every key of the proxy cache: what one build of the
    // layer kept must not be found by another, whose translator or rewriter may give something
    // else. It is a digest of the sources of every crate of the layer and of the versions of what
    // they use (`Cargo.lock`): this script runs again when any of them changes, so the identity
    // changes with the code and only with it. (It was the time this script last ran, and the
    // script ran again only when this crate changed: a build with another translator found the
    // translations of the one before.)
    let crates = Path::new(&dir).parent().unwrap();
    let mut sources = vec![crates.parent().unwrap().join("Cargo.lock")];
    let mut pending = vec![crates.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).unwrap().flatten() {
            let path = entry.path();
            let name = entry.file_name();
            if path.is_dir() {
                if name != "target" && name != "tests" {
                    pending.push(path);
                }
            } else if path
                .extension()
                .is_some_and(|e| e == "rs" || e == "toml" || e == "txt" || e == "tsv")
            {
                sources.push(path);
            }
        }
    }
    sources.sort();
    let mut digest = std::collections::hash_map::DefaultHasher::new();
    for path in &sources {
        use std::hash::Hasher;
        digest.write(
            path.strip_prefix(crates.parent().unwrap())
                .unwrap()
                .as_os_str()
                .as_encoded_bytes(),
        );
        digest.write(&fs::read(path).unwrap());
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let identity = {
        use std::hash::Hasher;
        digest.finish()
    };
    writeln!(out, "pub const BUILD_IDENTITY: u64 = {identity:#x};").unwrap();

    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("generated.rs"), out).unwrap();
    // The runtime opens the driver library under this name.
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libcuda.so.1");
    // (A file added to a crate: the directories too.)
    println!("cargo:rerun-if-changed={}", crates.display());
    println!("cargo:rerun-if-changed=symbols");
}
