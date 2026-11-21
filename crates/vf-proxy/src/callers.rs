//! Which compute capability a caller is told.
//!
//! A program chooses its kernels, and often its launch parameters, from the compute capability of the
//! GPU. Code built for sm_75 and later must be told a capability it has kernels for, or it gives up
//! (or launches kernels with the parameters of another architecture); once told 7.5 it loads its
//! Turing code, which `images` adapts. Code that does have Volta kernels must be told the truth, so
//! that it uses them: the CUDA 12 libraries that stand in for their CUDA 13 versions are in that case,
//! in the same process as the program built for CUDA 13.
//!
//! So the answer depends on who asks. The caller is the library the capability function returns to:
//! for example, a CUDA runtime, an application or a binding such as libffi.
//!
//! - A file that embeds code the driver uses on this GPU is told the real capability, always.
//! - `libcudart.so.13` and later, and a file whose embedded code starts at an architecture of the
//!   same generation that PTX can be retargeted from (sm_75 on Volta), are told that architecture.
//! - Under the launcher, a readable file with no embedded GPU code sees 7.5 for the device.
//!   Its queries about an individual kernel still return the architecture of the loaded binary.
//! - Older CUDA runtimes retain the physical capability by default.
//!
//! `VF_CAPABILITY` overrides the device answer for a direct client or code targeting later GPUs.
//! The code taken from the program follows that setting (`vf_adapt::Policy::told_sm`).

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::sync::{LazyLock, Mutex};

use vf_adapt::Embedded;

use crate::{config, log};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Caller {
    /// Has code for this GPU.
    Native,
    /// Has code from architecture `sm` on, which can be adapted.
    From(u32),
    /// A launcher client whose file contains no GPU code.
    Direct,
    /// Neither is known.
    Other,
}

/// First CUDA release without Volta: its runtime serves programs built for sm_75 and later.
const FIRST_RUNTIME_WITHOUT_DEVICE: u32 = 13;
const FIRST_SM_OF_THAT_RUNTIME: u32 = vf_adapt::TRANSLATED_FROM;

static UNDER_LAUNCHER: LazyLock<bool> = LazyLock::new(|| std::env::var_os("VOLTA_FOREVER").is_some());

#[derive(Default)]
struct Cache {
    unloads: u64,
    by_base: HashMap<usize, Caller>,
}

static CALLERS: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(Cache::default()));

/// glibc's unload counter changes even if another DSO takes the exact same address and filename.
fn unload_count() -> u64 {
    unsafe extern "C" fn first(info: *mut libc::dl_phdr_info, _: usize, data: *mut c_void) -> i32 {
        // SAFETY: glibc on the supported x86-64 Linux systems supplies dlpi_subs; data points to n.
        unsafe { *data.cast::<u64>() = (*info).dlpi_subs };
        1
    }
    let mut n = 0u64;
    unsafe { libc::dl_iterate_phdr(Some(first), (&raw mut n).cast()) };
    n
}

/// A file mapped read-only.
struct Mapped {
    start: *mut c_void,
    len: usize,
}

impl Mapped {
    fn open(path: &CStr) -> Option<Mapped> {
        // SAFETY: plain system calls on a path; the mapping is private and read-only.
        unsafe {
            let fd = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
            if fd < 0 {
                return None;
            }
            let mut stat: libc::stat = std::mem::zeroed();
            let len = if libc::fstat(fd, &mut stat) == 0 {
                stat.st_size as usize
            } else {
                0
            };
            let start = if len > 0 {
                libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, fd, 0)
            } else {
                libc::MAP_FAILED
            };
            libc::close(fd);
            (start != libc::MAP_FAILED).then_some(Mapped { start, len })
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers `len` readable bytes until `drop`.
        unsafe { std::slice::from_raw_parts(self.start as *const u8, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: `start` and `len` describe the mapping made in `open`.
        unsafe { libc::munmap(self.start, self.len) };
    }
}

/// The major version in `libcudart.so.<major>[.minor...]`.
fn runtime_version(file_name: &str) -> Option<u32> {
    file_name.strip_prefix("libcudart.so.")?.split('.').next()?.parse().ok()
}

fn classify(path: &CStr) -> Caller {
    let policy = &config().policy;
    let name = path.to_string_lossy();
    let file_name = name.rsplit('/').next().unwrap_or(&name);
    if let Some(version) = runtime_version(file_name) {
        return if version >= FIRST_RUNTIME_WITHOUT_DEVICE && policy.retargetable(FIRST_SM_OF_THAT_RUNTIME) {
            Caller::From(FIRST_SM_OF_THAT_RUNTIME)
        } else {
            Caller::Other
        };
    }
    match Mapped::open(path).map(|file| policy.embedded(file.bytes())) {
        Some(Embedded::Usable) => Caller::Native,
        Some(Embedded::Nothing) if *UNDER_LAUNCHER => Caller::Direct,
        Some(Embedded::Other { lowest_sm }) if policy.retargetable(lowest_sm) => Caller::From(lowest_sm),
        _ => Caller::Other,
    }
}

fn caller(return_address: usize) -> Caller {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: the call site is live for the duration of this driver call.
    if unsafe { libc::dladdr(return_address as *const c_void, &mut info) } == 0 || info.dli_fname.is_null() {
        return Caller::Other;
    }
    let unloads = unload_count();
    let mut cache = CALLERS.lock().unwrap_or_else(|e| e.into_inner());
    if cache.unloads != unloads {
        cache.by_base.clear();
        cache.unloads = unloads;
    }
    *cache.by_base.entry(info.dli_fbase as usize).or_insert_with(|| {
        let path = unsafe { CStr::from_ptr(info.dli_fname) };
        let caller = classify(path);
        let answer = match (caller, config().capability) {
            (Caller::Native, _) => "the real compute capability: it has code for this GPU".to_string(),
            (_, Some((major, minor))) => format!("compute capability {major}.{minor}, as VF_CAPABILITY asks"),
            (Caller::From(sm), None) => format!("compute capability {}.{}: its code starts there", sm / 10, sm % 10),
            (Caller::Direct, None) => "compute capability 7.5: it has no embedded GPU code".to_string(),
            (Caller::Other, None) => "the real compute capability".to_string(),
        };
        log::info(format_args!("{} is told {answer}", path.to_string_lossy()));
        caller
    })
}

/// The capability reported for a physical sm_70 device.
pub fn device_capability(return_address: usize) -> Option<(i32, i32)> {
    capability(caller(return_address))
}

/// A client without embedded GPU code can identify the native architecture of an external kernel.
pub fn binary_capability(return_address: usize) -> Option<(i32, i32)> {
    match caller(return_address) {
        Caller::Direct => None,
        other => capability(other),
    }
}

fn capability(caller: Caller) -> Option<(i32, i32)> {
    let sm = match caller {
        Caller::Native => return None,
        Caller::Direct => FIRST_SM_OF_THAT_RUNTIME,
        Caller::From(sm) => sm,
        Caller::Other => return config().capability,
    };
    config().capability.or(Some(((sm / 10) as i32, (sm % 10) as i32)))
}
