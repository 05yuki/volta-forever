//! Images as programs hand them to the driver, and their replacements.
//!
//! A program gives the driver a pointer to a fat-binary wrapper, a fat binary, a cubin or PTX text, most
//! often without a length. `adapt` looks at what the pointer leads to and, when the image carries
//! nothing the driver can use on this GPU but can be adapted, builds the image to load instead.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{LazyLock, Mutex};

use vf_adapt::{Adapted, Emulated, Outcome, Refused, RefusedFunction};
use vf_fatbin as fatbin;

use crate::{config, kernels, log};

/// The 24-byte structure through which the CUDA runtime passes a fat binary.
#[repr(C)]
#[derive(Clone, Copy)]
struct Wrapper {
    magic: u32,
    version: u32,
    data: *const u8,
    /// Version 1: file name; version 2: array of further containers.
    extra: *const c_void,
}

/// The driver's error for PTX its compiler refuses.
pub const CUDA_ERROR_INVALID_PTX: i32 = 218;
/// Attempts at loading an image without the functions the compiler refuses in it. Its messages name
/// all of them at once unless it stops on one it cannot parse.
pub const REFUSAL_ROUNDS: usize = 8;
/// Size of the buffer given to the driver for the compiler's messages: room for thousands of them.
pub const ERROR_LOG_SIZE: usize = 1 << 20;

/// An adapted image, owned by this library for as long as the driver may read it, and what the
/// adaptation found about its kernels.
pub struct Replacement {
    data: Box<[u8]>,
    wrapper: Option<Box<Wrapper>>,
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
    /// How the code reached the device, for the profiler.
    pub origin: Option<kernels::Origin>,
    /// For a second translation at launch (`fit`): the cubin that this is the translation of, when
    /// a kernel of the translation takes more registers than its budget.
    pub original_image: Option<Vec<u8>>,
}

// SAFETY: the pointers inside `wrapper` lead to `data`, which the replacement owns, and to memory of
// the program that the driver was going to read anyway.
unsafe impl Send for Replacement {}

impl Replacement {
    fn new(data: Vec<u8>, wrapper: Option<Wrapper>) -> Replacement {
        let data = data.into_boxed_slice();
        let wrapper = wrapper.map(|w| {
            Box::new(Wrapper {
                data: data.as_ptr(),
                ..w
            })
        });
        Replacement {
            data,
            wrapper,
            refused: Vec::new(),
            emulated: Vec::new(),
            origin: None,
            original_image: None,
        }
    }

    /// The image itself, where `as_ptr` may lead to a wrapper: a fat binary or text.
    pub fn data_ptr(&self) -> *const c_void {
        self.data.as_ptr() as *const c_void
    }

    /// Kernels of this image that would trap, which nothing keeps from the program on the path the
    /// image takes (the linker).
    pub fn refused(&self) -> &[Refused] {
        &self.refused
    }

    /// Kernels that run emulated instructions.
    pub fn emulated(&self) -> &[Emulated] {
        &self.emulated
    }

    /// The image to load after the driver's compiler refused functions of this one, as its `log`
    /// says, and those functions. See `vf_adapt::Policy::without_refused`.
    pub fn without_refused(&self, log: &str) -> Option<(Replacement, Vec<RefusedFunction>)> {
        let retry = config().policy.without_refused(&self.data, log)?;
        let mut next = Replacement::new(retry.image, self.wrapper.as_deref().copied());
        next.refused = self.refused.iter().cloned().chain(retry.kernels).collect();
        next.emulated = self.emulated.clone();
        next.origin = self.origin;
        next.original_image = self.original_image.clone();
        Some((next, retry.functions))
    }

    /// What to pass to the driver in place of the program's pointer.
    pub fn as_ptr(&self) -> *const c_void {
        match &self.wrapper {
            Some(wrapper) => &**wrapper as *const Wrapper as *const c_void,
            None => self.data.as_ptr() as *const c_void,
        }
    }

    /// Size of the image in bytes (of the fat binary or the text, not of a wrapper).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Create a replacement from an owned byte vector (for sub-modules loaded lazily).
    pub fn from_vec(data: Vec<u8>) -> Replacement {
        Replacement::new(data, None)
    }

    /// What was loaded in the end in place of this replacement: `image`, in which `refused` kernels
    /// more cannot be given to the program.
    pub fn loaded_as(self, image: Vec<u8>, refused: Vec<Refused>) -> Replacement {
        let mut loaded = Replacement::new(image, None);
        loaded.refused = self.refused.into_iter().chain(refused).collect();
        loaded.emulated = self.emulated;
        loaded.origin = self.origin;
        loaded.original_image = self.original_image;
        loaded
    }

    /// The raw bytes of the image (fat binary or PTX text).
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Replacements in use, by the module or library handle the driver returned for them. The driver may
/// read an image again after the load returns (lazy loading), so they live until the unload.
static KEPT: LazyLock<Mutex<HashMap<usize, Replacement>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// The driver loaded `replacement` as `handle`: keep it, and what is known of its kernels.
pub fn keep(handle: usize, mut replacement: Replacement) {
    // Context destruction can retire a CUmodule without an explicit unload.
    // The driver's next load starts a new lifetime at the same address.
    kernels::forget(handle);
    if config().profile
        && let Some(origin) = replacement.origin
    {
        kernels::set_origin(handle, origin);
    }
    kernels::describe(
        handle,
        std::mem::take(&mut replacement.refused),
        std::mem::take(&mut replacement.emulated),
    );
    // For a second translation at launch (`fit`): the original of a translation that has a kernel
    // over its budget. Otherwise nothing: and nothing may be left there under a handle that the
    // driver gives again.
    match replacement.original_image.take() {
        Some(original) if !original.is_empty() => crate::fit::register_module(handle, original),
        _ => crate::fit::forget(handle),
    }
    KEPT.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(handle, replacement);
}

pub fn release(handle: usize) {
    KEPT.lock().unwrap_or_else(|e| e.into_inner()).remove(&handle);
    kernels::forget(handle);
}

fn report(who: &str, what: &str, outcome: &Outcome) {
    match outcome {
        Outcome::Native => log::trace(format_args!("{who}: {what} with code for this GPU")),
        Outcome::Unknown => log::trace(format_args!("{who}: {what} not recognised, loaded as is")),
        Outcome::Retargeted {
            entries,
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated: _,
        } => {
            let device = config().policy.device_sm;
            let mut changes = format!("{entries} {}", if *entries == 1 { "module" } else { "modules" });
            if *tensor_rewritten > 0 {
                changes += &format!(", {tensor_rewritten} tensor-core instructions rewritten");
            }
            if *tanh_expanded > 0 {
                changes += &format!(", {tanh_expanded} tanh expanded");
            }
            log::info(format_args!(
                "{who}: {what} without sm_{device} code: PTX for sm_{from_sm} retargeted ({changes})"
            ));
            if let Some(first) = trapped.first() {
                log::info(format_args!(
                    "{who}: {} functions use instructions without an sm_{device} sequence and were made to trap \
                     (the first is {}: {}); the {} kernels that need them are refused if the program asks for them",
                    trapped.len(),
                    first.function,
                    first.reason,
                    refused.len()
                ));
            }
        }
        Outcome::Translated {
            entries,
            from_sm,
            functions,
            refused,
            ..
        } => {
            let device = config().policy.device_sm;
            let modules = if *entries == 1 {
                String::new()
            } else {
                format!(" in {entries} modules")
            };
            log::info(format_args!(
                "{who}: {what} without sm_{device} code: machine code for sm_{from_sm} translated \
                 ({functions} functions{modules})"
            ));
            if let Some(first) = refused.first() {
                log::info(format_args!(
                    "{who}: {} of these functions {} not translated (the first is {}: {}); their kernels are \
                     refused if the program asks for them",
                    refused.len(),
                    if refused.len() == 1 { "is" } else { "are" },
                    first.function,
                    first.reason
                ));
            }
        }
        Outcome::NotTranslated { from_sm, reason } => log::warn(format_args!(
            "{who}: {what} with machine code for sm_{from_sm} that could not be read ({reason}); loaded as is"
        )),
        Outcome::NeedsTranslation { lowest_sm } => log::warn(format_args!(
            "{who}: {what} with machine code for sm_{lowest_sm} and later only, which nothing here \
             translates; loaded as is"
        )),
    }
}

thread_local! {
    /// Profiler only: the last `adapt` of this thread found code for this GPU in the image and
    /// left it as it is. Read by `loaded_as_is`, which labels the module the driver then loads.
    static IMAGE_WAS_NATIVE: Cell<bool> = const { Cell::new(false) };
}

/// Reset metadata after the program's own image was loaded. Also label native
/// images when profiling is enabled.
///
/// # Safety
/// `handle` must be null or point to the handle the driver stored.
pub unsafe fn loaded_as_is(result: c_int, handle: *mut *mut c_void) {
    if result == 0 && !handle.is_null() {
        // (Nothing of a module that is gone may be left under a handle that the driver gives again.)
        let fresh = unsafe { *handle } as usize;
        release(fresh);
        crate::fit::forget(fresh);
    }
    if !config().profile {
        return;
    }
    if IMAGE_WAS_NATIVE.replace(false) && result == 0 && !handle.is_null() {
        kernels::set_origin(unsafe { *handle } as usize, kernels::Origin::Native);
    }
}

/// `original`: the image as the program gave it, when it is machine code that may have been
/// translated (kept, for `fit`, when the translation has a kernel over its budget).
fn finish(
    who: &str,
    what: &str,
    adapted: Adapted,
    wrapper: Option<Wrapper>,
    terminate: bool,
    original: Option<&[u8]>,
) -> Option<Replacement> {
    report(who, what, &adapted.outcome);
    if config().profile && matches!(adapted.outcome, Outcome::Native) {
        IMAGE_WAS_NATIVE.set(true);
    }
    let over_budget = matches!(adapted.outcome, Outcome::Translated { over_budget: true, .. });
    let origin = match &adapted.outcome {
        Outcome::Retargeted { .. } => Some(kernels::Origin::PtxRewritten),
        Outcome::Translated { .. } => Some(kernels::Origin::MachineCodeTranslated),
        _ => None,
    };
    let mut image = adapted.image?;
    if terminate {
        image.push(0);
    }
    let mut replacement = Replacement::new(image, wrapper);
    replacement.origin = origin;
    if over_budget
        && crate::fit::enabled()
        && let Some(original) = original
    {
        // Kept only when a second translation within the budget would lower a kernel's count (the
        // translation says so, and the cache keeps it said), and then the one cubin translated,
        // as it is stored, not the fat binary around it.
        replacement.original_image = config().policy.kept_for_second(original);
    }
    if let Outcome::Retargeted { refused, emulated, .. } | Outcome::Translated { refused, emulated, .. } =
        adapted.outcome
    {
        replacement.refused = refused;
        replacement.emulated = emulated;
    }
    Some(replacement)
}

/// What `adapt` makes of `image`, a whole fat binary or cubin. The translation of machine code
/// takes a tenth of a second a megabyte: it is kept (`cache`), and taken from there when the same
/// image was translated by an earlier run of the same build with the same policy.
fn adapted(who: &str, image: &[u8], adapt: impl FnOnce() -> Adapted) -> Adapted {
    let key = crate::cache::enabled().then(|| crate::cache::key_for(image));
    let start = std::time::Instant::now();
    if let Some(kept) = key.as_ref().and_then(crate::cache::load_translation) {
        log::timing(format_args!(
            "timing: {who}: translation from the cache {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        ));
        return kept;
    }
    let adapted = adapt();
    if let Outcome::Translated { .. } = adapted.outcome {
        log::timing(format_args!(
            "timing: {who}: machine code translated in {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        ));
        if let Some(key) = &key {
            crate::cache::save_translation(key, &adapted);
        }
    }
    adapted
}

/// `at`: where the fat binary is, for the bytes of the whole of it.
fn adapt_fatbin(
    who: &str,
    at: *const u8,
    fat: Result<fatbin::FatBinary<'_>, fatbin::Error>,
    wrapper: Option<Wrapper>,
) -> Option<Replacement> {
    // SAFETY: `fat` was read from `at`, and is `fat.len` bytes there.
    let original = fat
        .as_ref()
        .ok()
        .map(|fat| unsafe { std::slice::from_raw_parts(at, fat.len) });
    let adapted = fat.and_then(|fat| {
        // SAFETY: as above.
        let whole = unsafe { std::slice::from_raw_parts(at, fat.len) };
        let mut failed = None;
        let adapted = adapted(who, whole, || {
            config().policy.adapt_fatbin(&fat).unwrap_or_else(|e| {
                failed = Some(e);
                Adapted {
                    outcome: Outcome::Unknown,
                    image: None,
                }
            })
        });
        failed.map_or(Ok(adapted), Err)
    });
    match adapted {
        Ok(adapted) => finish(who, "fat binary", adapted, wrapper, false, original),
        Err(e) => {
            log::warn(format_args!("{who}: {e}; loaded as is"));
            None
        }
    }
}

/// Copy the original image bytes at `ptr` exactly as the adaptation would consume them: the fat
/// binary a wrapper points to, the fat binary at `ptr`, or the PTX text (until the terminating zero).
/// Returns `None` for a null pointer, for machine code, for a format the adaptation would not touch,
/// or when adaptation is off.
///
/// # Safety
/// `ptr` must satisfy the same contract as `adapt`.
pub unsafe fn image_bytes(ptr: *const c_void, len: Option<usize>) -> Option<Vec<u8>> {
    if !config().adapt || ptr.is_null() {
        return None;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { image_bytes_inner(ptr, len) }))
        .ok()
        .flatten()
}

/// # Safety
/// As for `image_bytes`.
unsafe fn image_bytes_inner(ptr: *const c_void, len: Option<usize>) -> Option<Vec<u8>> {
    let bytes = ptr as *const u8;
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let binary = magic == fatbin::WRAPPER_MAGIC || magic == fatbin::MAGIC || head == *b"\x7fELF";
    if !binary && (head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        let text = match len {
            Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
            None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
        };
        return Some(text.to_vec());
    }
    if magic == fatbin::WRAPPER_MAGIC {
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        if wrapper.version != 1 || wrapper.data.is_null() {
            return None;
        }
        let container_len =
            fatbin::container_len(unsafe { std::slice::from_raw_parts(wrapper.data, fatbin::HEADER_SIZE) }).ok()?;
        return Some(unsafe { std::slice::from_raw_parts(wrapper.data, container_len) }.to_vec());
    }
    if magic == fatbin::MAGIC {
        let container_len =
            fatbin::container_len(unsafe { std::slice::from_raw_parts(bytes, fatbin::HEADER_SIZE) }).ok()?;
        return Some(unsafe { std::slice::from_raw_parts(bytes, container_len) }.to_vec());
    }
    // Machine code (an ELF file): how far it reaches is not known here without reading its section
    // table, and its header alone does not tell it from another. It is not cached.
    None
}

/// # Safety
/// As for `adapt`.
unsafe fn adapt_unguarded(who: &str, ptr: *const c_void, len: Option<usize>) -> Option<Replacement> {
    let bytes = ptr as *const u8;
    let policy = &config().policy;
    // The first four bytes tell the binary formats apart. Text may be shorter than that: stop at its
    // terminating zero. (A fat binary starts with 0x50, the letter P, so text is what remains.)
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let binary = magic == fatbin::WRAPPER_MAGIC || magic == fatbin::MAGIC || head == *b"\x7fELF";
    if !binary && (head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        let text = match len {
            Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
            None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
        };
        // Text given with a length is not zero-terminated; neither is its replacement.
        return finish(who, "PTX text", policy.adapt_ptx(text), None, len.is_none(), None);
    }
    if magic == fatbin::WRAPPER_MAGIC {
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        if wrapper.version != 1 || wrapper.data.is_null() {
            log::warn(format_args!(
                "{who}: fat binary wrapper version {} loaded as is",
                wrapper.version
            ));
            return None;
        }
        return adapt_fatbin(
            who,
            wrapper.data,
            unsafe { fatbin::parse_at(wrapper.data) },
            Some(wrapper),
        );
    }
    if magic == fatbin::MAGIC {
        return adapt_fatbin(who, bytes, unsafe { fatbin::parse_at(bytes) }, None);
    }
    if head == *b"\x7fELF" {
        let header = unsafe { std::slice::from_raw_parts(bytes, 0x40) };
        if !policy.translates_cubin(header) {
            return finish(who, "cubin", policy.adapt_cubin(header), None, false, None);
        }
        // To be translated, all of it: how far it reaches is in its tables, which the driver would
        // read as well.
        let whole = fatbin::cubin_tables_len(header)
            .and_then(|tables| fatbin::cubin_len(unsafe { std::slice::from_raw_parts(bytes, tables) }));
        let Some(length) = whole else {
            log::warn(format_args!(
                "{who}: cubin with machine code to translate, whose length its tables do not tell; loaded as is"
            ));
            return None;
        };
        let cubin = unsafe { std::slice::from_raw_parts(bytes, length) };
        return finish(
            who,
            "cubin",
            adapted(who, cubin, || policy.adapt_cubin(cubin)),
            None,
            false,
            Some(cubin),
        );
    }
    log::trace(format_args!(
        "{who}: image with unknown magic {magic:#010x} loaded as is"
    ));
    None
}

/// The image to load in place of the one at `ptr`, if it has to be adapted. `len` is its length when
/// the caller states one (`cuLinkAddData`).
///
/// # Safety
/// `ptr` must be null or point to an image the driver itself could read: a complete fat-binary wrapper,
/// fat binary or cubin, or text ended by a zero byte (or `len` bytes long).
pub unsafe fn adapt(who: &str, ptr: *const c_void, len: Option<usize>) -> Option<Replacement> {
    if config().profile {
        IMAGE_WAS_NATIVE.set(false);
    }
    if !config().adapt || ptr.is_null() {
        return None;
    }
    // Whatever goes wrong here, the program must get what it would get without this library.
    catch_unwind(AssertUnwindSafe(|| unsafe { adapt_unguarded(who, ptr, len) })).unwrap_or_else(|_| {
        log::error(format_args!(
            "{who}: internal error while adapting an image; loaded as is"
        ));
        None
    })
}

/// The image to load in place of the contents of a file, if it has to be adapted.
pub fn adapt_file(who: &str, path: &CStr) -> Option<Replacement> {
    if config().profile {
        IMAGE_WAS_NATIVE.set(false);
    }
    if !config().adapt {
        return None;
    }
    let mut contents = std::fs::read(path.to_str().ok()?).ok()?;
    if contents.len() < 4 {
        return None;
    }
    if contents.starts_with(&fatbin::MAGIC.to_le_bytes()) {
        // A file cut short must not be read past its end.
        fatbin::parse(&contents).ok()?;
    } else if fatbin::is_elf(&contents) {
        // A file cut short must not be read past its end either.
        if contents.len() < 0x40 || fatbin::cubin_len(&contents).is_none_or(|length| length > contents.len()) {
            return None;
        }
    } else if contents.starts_with(&fatbin::WRAPPER_MAGIC.to_le_bytes()) {
        return None; // a wrapper holds pointers: it has no meaning in a file
    } else {
        contents.push(0); // text
    }
    // SAFETY: `contents` is a complete fat binary, a complete cubin, or zero-terminated bytes.
    unsafe { adapt(who, contents.as_ptr().cast(), None) }
}
