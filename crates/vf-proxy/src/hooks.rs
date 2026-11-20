//! The driver functions this library answers itself instead of forwarding blindly: the lookups of the
//! CUDA runtime, the device capability, and every function that takes GPU code.
//!
//! `build.rs` finds the functions defined here by their `extern "C" fn <name>(` text and generates a
//! plain jump for all the others.
#![allow(non_snake_case)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::Ordering::Relaxed;

use crate::images::{CUDA_ERROR_INVALID_PTX, Replacement};
use crate::{COUNTS, REALS, callers, config, fit, idx, images, kernels, lazy, log, own_entry, profile};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
/// `CUjit_option` values of the buffer for the compiler's error messages and of its size.
const JIT_ERROR_LOG_BUFFER: c_int = 5;
const JIT_ERROR_LOG_BUFFER_SIZE: c_int = 6;
/// `CUdevice_attribute` values of the compute capability.
const ATTRIBUTE_CAPABILITY_MAJOR: c_int = 75;
const ATTRIBUTE_CAPABILITY_MINOR: c_int = 76;
/// `CUjitInputType` values.
const INPUT_PTX: c_int = 1;
const INPUT_FATBINARY: c_int = 2;

/// NVIDIA's implementation of a function, as a pointer of the given type; the call is counted.
macro_rules! real {
    ($name:ident: $ty:ty) => {{
        COUNTS[idx::$name].fetch_add(1, Relaxed);
        // SAFETY: REALS holds the address of NVIDIA's function of that name, whose C prototype is
        // `$ty`, or of a stub that ignores its arguments and returns an error code.
        unsafe { std::mem::transmute::<usize, $ty>(REALS[idx::$name].load(Relaxed)) }
    }};
}

// ---- lookups ---------------------------------------------------------------------------------------

/// Answer a successful lookup of a hooked function with this library's entry point, so that the calls
/// the runtime makes through the result (including its later lookups) come here.
unsafe fn interpose(symbol: *const c_char, function: *mut *mut c_void, version: c_int, result: CUresult) {
    if result != CUDA_SUCCESS || function.is_null() {
        return;
    }
    if let Some(own) = own_entry(unsafe { *function } as usize) {
        unsafe { *function = own as *mut c_void };
        if log::tracing() && !symbol.is_null() {
            let name = unsafe { CStr::from_ptr(symbol) }.to_string_lossy();
            log::trace(format_args!(
                "cuGetProcAddress({name}, {version}) answered with this library's entry"
            ));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress(
    symbol: *const c_char,
    function: *mut *mut c_void,
    version: c_int,
    flags: u64,
) -> CUresult {
    let real = real!(cuGetProcAddress: unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64) -> CUresult);
    let result = unsafe { real(symbol, function, version, flags) };
    unsafe { interpose(symbol, function, version, result) };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    function: *mut *mut c_void,
    version: c_int,
    flags: u64,
    status: *mut c_void,
) -> CUresult {
    let real = real!(cuGetProcAddress_v2:
        unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64, *mut c_void) -> CUresult);
    let result = unsafe { real(symbol, function, version, flags, status) };
    unsafe { interpose(symbol, function, version, result) };
    result
}

// ---- device capability -----------------------------------------------------------------------------
//
// The answer depends on the caller (`callers`): the exported functions jump to their body with the
// address they will return to as one more argument.

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> CUresult {
    core::arch::naked_asm!("mov rcx, [rsp]", "jmp {body}", body = sym device_attribute)
}

unsafe extern "C" fn device_attribute(value: *mut c_int, attribute: c_int, device: c_int, caller: usize) -> CUresult {
    let real = real!(cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult);
    let result = unsafe { real(value, attribute, device) };
    if result != CUDA_SUCCESS || value.is_null() {
        return result;
    }
    if !matches!(attribute, ATTRIBUTE_CAPABILITY_MAJOR | ATTRIBUTE_CAPABILITY_MINOR) {
        return result;
    }
    if let Some((major, minor)) = callers::capability(caller) {
        unsafe {
            *value = if attribute == ATTRIBUTE_CAPABILITY_MAJOR {
                major
            } else {
                minor
            }
        };
    }
    result
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, device: c_int) -> CUresult {
    core::arch::naked_asm!("mov rcx, [rsp]", "jmp {body}", body = sym device_capability)
}

unsafe extern "C" fn device_capability(major: *mut c_int, minor: *mut c_int, device: c_int, caller: usize) -> CUresult {
    let real = real!(cuDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> CUresult);
    let result = unsafe { real(major, minor, device) };
    if result != CUDA_SUCCESS {
        return result;
    }
    if let Some(capability) = callers::capability(caller) {
        if !major.is_null() {
            unsafe { *major = capability.0 };
        }
        if !minor.is_null() {
            unsafe { *minor = capability.1 };
        }
    }
    result
}

// ---- the architecture of a kernel's machine code ----------------------------------------------------
//
// A program may ask which architecture the machine code of one of its kernels is for
// (`cudaFuncGetAttributes`, `binaryVersion`), to see that the driver took the code it meant: the
// backward pass of PyTorch's attention stops when the answer is below the architecture of the
// variant it chose. What the driver has is the layer's code for this GPU in place of the
// program's, and it answers 7.0. A caller that is told a capability is answered at least that.

/// `CUfunction_attribute` value of the architecture of a function's machine code: major * 10 + minor.
const FUNCTION_BINARY_VERSION: c_int = 6;

/// After NVIDIA's answer `result` to a question about a function: the architecture of its machine
/// code is never below the capability `caller` is told.
unsafe fn binary_version(value: *mut c_int, attribute: c_int, result: CUresult, caller: usize) {
    if result != CUDA_SUCCESS || value.is_null() || attribute != FUNCTION_BINARY_VERSION {
        return;
    }
    if let Some((major, minor)) = callers::capability(caller) {
        let told = major * 10 + minor;
        if unsafe { *value } < told {
            unsafe { *value = told };
        }
    }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncGetAttribute(value: *mut c_int, attribute: c_int, function: *mut c_void) -> CUresult {
    core::arch::naked_asm!("mov rcx, [rsp]", "jmp {body}", body = sym function_attribute)
}

unsafe extern "C" fn function_attribute(
    value: *mut c_int,
    attribute: c_int,
    function: *mut c_void,
    caller: usize,
) -> CUresult {
    let real = real!(cuFuncGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> CUresult);
    let result = unsafe { real(value, attribute, function) };
    unsafe { binary_version(value, attribute, result, caller) };
    result
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelGetAttribute(
    value: *mut c_int,
    attribute: c_int,
    kernel: *mut c_void,
    device: c_int,
) -> CUresult {
    core::arch::naked_asm!("mov r8, [rsp]", "jmp {body}", body = sym kernel_attribute)
}

unsafe extern "C" fn kernel_attribute(
    value: *mut c_int,
    attribute: c_int,
    kernel: *mut c_void,
    device: c_int,
    caller: usize,
) -> CUresult {
    let real = real!(cuKernelGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void, c_int) -> CUresult);
    let result = unsafe { real(value, attribute, kernel, device) };
    unsafe { binary_version(value, attribute, result, caller) };
    result
}

// ---- code loading ------------------------------------------------------------------------------------

type ModuleLoadEx =
    unsafe extern "C" fn(*mut *mut c_void, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CUresult;

/// Load `image` into `module` with `cuModuleLoadDataEx` and the program's options, and return the
/// compiler's error messages with the result. They are written to the program's own buffer when its
/// options name one, else to one added to the options for the call. What the driver writes back into
/// the options goes to the program.
///
/// # Safety
/// `module` must be writable; `options` and `values` null or valid for `count` elements, as for the
/// driver; `image` an image the driver can read.
unsafe fn load_logged(
    module: *mut *mut c_void,
    image: *const c_void,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> (CUresult, String) {
    let real = real!(cuModuleLoadDataEx: ModuleLoadEx);
    let given = if options.is_null() || values.is_null() {
        0
    } else {
        count as usize
    };
    let mut all_options: Vec<c_int> = (0..given).map(|i| unsafe { *options.add(i) }).collect();
    let mut all_values: Vec<*mut c_void> = (0..given).map(|i| unsafe { *values.add(i) }).collect();
    let position = |option: c_int| all_options.iter().position(|&o| o == option);
    let mut own: Vec<u8> = Vec::new();
    let (buffer, size) = match (position(JIT_ERROR_LOG_BUFFER), position(JIT_ERROR_LOG_BUFFER_SIZE)) {
        (Some(buffer), Some(size)) => (all_values[buffer] as *const u8, all_values[size] as usize),
        (None, None) => {
            own.resize(images::ERROR_LOG_SIZE, 0);
            all_options.extend([JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE]);
            all_values.extend([own.as_mut_ptr() as *mut c_void, own.len() as *mut c_void]);
            (own.as_ptr(), own.len())
        }
        // Half a buffer: nothing can be read back.
        _ => (std::ptr::null(), 0),
    };
    let result = unsafe {
        real(
            module,
            image,
            all_options.len() as c_uint,
            all_options.as_mut_ptr(),
            all_values.as_mut_ptr(),
        )
    };
    for (i, value) in all_values.iter().take(given).enumerate() {
        unsafe { *values.add(i) = *value };
    }
    if buffer.is_null() || size == 0 {
        return (result, String::new());
    }
    // SAFETY: `size` bytes that are this function's own, or that the program told the driver to
    // write to.
    let written = unsafe { std::slice::from_raw_parts(buffer, size) };
    let length = written.iter().position(|&byte| byte == 0).unwrap_or(size);
    (result, String::from_utf8_lossy(&written[..length]).into_owned())
}

/// Load the adapted image `replacement` with `call`, as the program asked. When the driver's compiler
/// refuses functions of it, load it again without them, through `logged`, which also returns the
/// compiler's messages. `Err` is the driver's refusal when nothing could be loaded.
///
/// # Safety
/// `handle` must be null or writable, and is where `call` and `logged` store the module handle.
unsafe fn load_adapted(
    who: &str,
    handle: *mut *mut c_void,
    mut replacement: Replacement,
    call: impl Fn(*const c_void) -> CUresult,
    logged: impl Fn(*const c_void) -> (CUresult, String),
) -> Result<(), CUresult> {
    let result = call(replacement.as_ptr());
    if result == CUDA_SUCCESS {
        if !handle.is_null() {
            images::keep(unsafe { *handle } as usize, replacement);
        }
        return Ok(());
    }
    if result != CUDA_ERROR_INVALID_PTX || handle.is_null() {
        return Err(result);
    }
    // The compiler refused something of the adapted image; which functions, only its messages tell.
    for _ in 0..images::REFUSAL_ROUNDS {
        let (again, messages) = logged(replacement.data_ptr());
        if again == CUDA_SUCCESS {
            images::keep(unsafe { *handle } as usize, replacement);
            return Ok(());
        }
        if again != CUDA_ERROR_INVALID_PTX {
            return Err(again);
        }
        let (next, functions) = replacement.without_refused(&messages).ok_or(again)?;
        log::info(format_args!(
            "{who}: the compiler refuses {} functions of the adapted image (the first is {}: {}); loading it \
             without them",
            functions.len(),
            functions[0].name,
            functions[0].reasons[0]
        ));
        replacement = next;
    }
    Err(result)
}

/// Load `image` with `call`, adapted if needed (see `load_adapted`). When the driver refuses the
/// adapted image whatever is done, the original is loaded, so that the program gets the result it
/// would get without this library.
///
/// # Safety
/// `image` must satisfy `images::adapt`; `handle` must be null or writable, and is where `call` and
/// `logged` store the module handle.
unsafe fn load(
    who: &str,
    handle: *mut *mut c_void,
    image: *const c_void,
    call: impl Fn(*const c_void) -> CUresult,
    logged: impl Fn(*const c_void) -> (CUresult, String),
) -> CUresult {
    let Some(replacement) = (unsafe { images::adapt(who, image, None) }) else {
        let result = call(image);
        unsafe { images::loaded_as_is(result, handle) };
        return result;
    };
    match unsafe { load_adapted(who, handle, replacement, &call, logged) } {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "{who}: the driver refused the adapted image (error {result}); loading the original"
            ));
            call(image)
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> CUresult {
    let real = real!(cuModuleLoadData: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult);
    unsafe {
        load(
            "cuModuleLoadData",
            module,
            image,
            |image| real(module, image),
            |image| load_logged(module, image, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadDataEx(
    module: *mut *mut c_void,
    image: *const c_void,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuModuleLoadDataEx: ModuleLoadEx);
    unsafe {
        load(
            "cuModuleLoadDataEx",
            module,
            image,
            |image| real(module, image, count, options, values),
            |image| load_logged(module, image, count, options, values),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadFatBinary(module: *mut *mut c_void, image: *const c_void) -> CUresult {
    let real = real!(cuModuleLoadFatBinary: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult);
    unsafe {
        load(
            "cuModuleLoadFatBinary",
            module,
            image,
            |image| real(module, image),
            |image| load_logged(module, image, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoad(module: *mut *mut c_void, path: *const c_char) -> CUresult {
    let real = real!(cuModuleLoad: unsafe extern "C" fn(*mut *mut c_void, *const c_char) -> CUresult);
    let replacement = if path.is_null() {
        None
    } else {
        images::adapt_file("cuModuleLoad", unsafe { CStr::from_ptr(path) })
    };
    let Some(replacement) = replacement else {
        let result = unsafe { real(module, path) };
        unsafe { images::loaded_as_is(result, module) };
        return result;
    };
    let from_data = real!(cuModuleLoadData: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult);
    let loaded = unsafe {
        load_adapted(
            "cuModuleLoad",
            module,
            replacement,
            |image| from_data(module, image),
            |image| load_logged(module, image, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        )
    };
    match loaded {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "cuModuleLoad: the driver refused the adapted image (error {result}); loading the file"
            ));
            unsafe { real(module, path) }
        }
    }
}

/// A lookup of kernel `name` in the module or library `handle`: refused when the kernel cannot run on
/// this GPU, else done by `find`; a kernel that is found and runs emulated instructions is announced.
/// When `kernel_out` is not null and the lookup succeeds, the profiler records the kernel handle ->
/// name mapping (since `cuFuncGetName` does not work for lazily compiled sub-modules).
///
/// # Safety
/// `name` must be null or a C string. `kernel_out` must be null or point to the handle the driver
/// stored.
unsafe fn lookup(
    handle: usize,
    name: *const c_char,
    kernel_out: *mut *mut c_void,
    find: impl FnOnce() -> CUresult,
) -> CUresult {
    if name.is_null() {
        return find();
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    if kernels::refused(handle, &name) {
        return CUDA_ERROR_INVALID_PTX;
    }
    let result = find();
    if result == CUDA_SUCCESS {
        kernels::given(handle, &name);
        // Record the kernel handle -> name mapping for the profiler: cuFuncGetName does not
        // work for lazily compiled sub-modules, so the profiler uses this map instead.
        if config().profile && !kernel_out.is_null() {
            let func_handle = unsafe { *kernel_out } as usize;
            if func_handle != 0 {
                profile::record_handle_name(handle, func_handle, &name);
            }
        }
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetFunction(
    function: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetFunction: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { lookup(module as usize, name, function, || real(function, module, name)) };
    if result == CUDA_SUCCESS && !function.is_null() && !name.is_null() {
        let func_handle = unsafe { *function } as usize;
        let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        fit::register_kernel(func_handle, module as usize, &name_str, false);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleUnload(module: *mut c_void) -> CUresult {
    let real = real!(cuModuleUnload: unsafe extern "C" fn(*mut c_void) -> CUresult);
    let result = unsafe { real(module) };
    fit::unload(module as usize);
    images::release(module as usize);
    result
}

// ---- globals and textures: mark modules from which the program accessed these -----------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetGlobal(
    dptr: *mut u64,
    size: *mut usize,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetGlobal: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { real(dptr, size, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetGlobal_v2(
    dptr: *mut u64,
    size: *mut usize,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetGlobal_v2: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { real(dptr, size, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetTexRef(
    tex_ref: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real = real!(cuModuleGetTexRef: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { real(tex_ref, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetSurfRef(
    surf_ref: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetSurfRef: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { real(surf_ref, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

// ---- function attributes: recorded so they can be replayed on the second-translation kernel ---------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncSetAttribute(function: *mut c_void, attribute: c_int, value: c_int) -> CUresult {
    let real = real!(cuFuncSetAttribute: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult);
    fit::set_attribute(
        function as usize,
        fit::Attribute::Function(attribute, value),
        || unsafe { real(function, attribute, value) },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncSetCacheConfig(function: *mut c_void, config: c_int) -> CUresult {
    let real = real!(cuFuncSetCacheConfig: unsafe extern "C" fn(*mut c_void, c_int) -> CUresult);
    fit::set_attribute(function as usize, fit::Attribute::FunctionCache(config), || unsafe {
        real(function, config)
    })
}

// Header: CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelSetAttribute(
    attribute: c_int,
    value: c_int,
    kernel: *mut c_void,
    device: c_int,
) -> CUresult {
    let real = real!(cuKernelSetAttribute: unsafe extern "C" fn(c_int, c_int, *mut c_void, c_int) -> CUresult);
    fit::set_attribute(
        kernel as usize,
        fit::Attribute::Kernel(attribute, value, device),
        || unsafe { real(attribute, value, kernel, device) },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelSetCacheConfig(kernel: *mut c_void, config: c_int, device: c_int) -> CUresult {
    let real = real!(cuKernelSetCacheConfig: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult);
    fit::set_attribute(
        kernel as usize,
        fit::Attribute::KernelCache(config, device),
        || unsafe { real(kernel, config, device) },
    )
}

type LibraryLoad<Source> = unsafe extern "C" fn(
    *mut *mut c_void,
    Source,
    *mut c_int,
    *mut *mut c_void,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
    c_uint,
) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryLoadData(
    library: *mut *mut c_void,
    code: *const c_void,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> CUresult {
    let real = real!(cuLibraryLoadData: LibraryLoad<*const c_void>);
    let call = |code: *const c_void| unsafe {
        real(
            library,
            code,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };

    // Proxy cache: hash the original image before adapting it.  On a cache hit the adaptation
    // and analysis are skipped entirely.  A copy of the original bytes is kept so that the
    // fallback path (materialization, dependent entries) can adapt later.
    let original_bytes = if lazy::enabled() && crate::cache::enabled() {
        unsafe { images::image_bytes(code, None) }
    } else {
        None
    };
    let cache_key = original_bytes.as_deref().map(crate::cache::key_for);

    if lazy::enabled()
        && let Some(ref ck) = cache_key
        && let Some(manifest) = crate::cache::load_manifest(ck)
    {
        let t0 = std::time::Instant::now();
        let result = unsafe {
            lazy::setup_from_cache(
                library,
                &manifest,
                // Keep the original bytes: the program may free `code` after the load
                // returns, and the fallback path needs them to adapt.
                original_bytes.clone().unwrap(),
                crate::cache::CacheKey { key: ck.key },
                jit_options,
                jit_values,
                jit_count,
                library_options,
                library_values,
                library_count,
            )
        };
        let t_cache = t0.elapsed();
        if result {
            log::timing(format_args!(
                "timing: cuLibraryLoadData: cache hit {:.1} ms, {} entries",
                t_cache.as_secs_f64() * 1000.0,
                manifest.entries.len(),
            ));
            if config().profile && !library.is_null() {
                kernels::set_origin(unsafe { *library } as usize, kernels::Origin::CacheHit);
            }
            return CUDA_SUCCESS;
        }
        // Cache hit but setup failed (e.g. single-entry module, stand-in refused):
        // fall through to adaptation.
    }

    // Adapt the image (returns None when adaptation is off, when the pointer is null, or when the
    // image already has code the driver can use).
    let t_adapt_start = std::time::Instant::now();
    let Some(replacement) = (unsafe { images::adapt("cuLibraryLoadData", code, None) }) else {
        let result = call(code);
        unsafe { images::loaded_as_is(result, library) };
        return result;
    };
    let t_adapt = t_adapt_start.elapsed();
    log::timing(format_args!(
        "timing: cuLibraryLoadData: adapt {:.1} ms, adapted image {} bytes",
        t_adapt.as_secs_f64() * 1000.0,
        replacement.len(),
    ));

    // Try lazy compilation: defer JIT of a multi-entry PTX module.
    if lazy::enabled() {
        let adapted_data = replacement.data_bytes().to_vec();
        if unsafe {
            lazy::setup(
                library,
                adapted_data,
                replacement.refused(),
                replacement.emulated(),
                jit_options,
                jit_values,
                jit_count,
                library_options,
                library_values,
                library_count,
                cache_key,
            )
        } {
            // Lazy setup succeeded: the stand-in handle is in *library. Keep the replacement so
            // the driver can lazy-read the stand-in image.
            if !library.is_null() {
                images::keep(unsafe { *library } as usize, replacement);
            }
            return CUDA_SUCCESS;
        }
    }

    // Load the adapted image through the driver.
    let loaded = unsafe {
        load_library(
            library,
            replacement,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    match loaded {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "cuLibraryLoadData: the driver refused the adapted image (error {result}); loading the original"
            ));
            call(code)
        }
    }
}

/// Load the adapted image of a library at once, without the functions the compiler refuses in it
/// (`lazy::load_now`), and store its handle in `library`.
///
/// # Safety
/// The arguments of the program's `cuLibraryLoadData` or `cuLibraryLoadFromFile`.
#[allow(clippy::too_many_arguments)]
unsafe fn load_library(
    library: *mut *mut c_void,
    replacement: Replacement,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> Result<(), CUresult> {
    let image = replacement.data_bytes().to_vec();
    let (handle, image, refused) = unsafe {
        lazy::load_now(
            image,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    }?;
    if library.is_null() {
        // Nowhere to give the handle: what the driver answers to that.
        let unload = real!(cuLibraryUnload: unsafe extern "C" fn(*mut c_void) -> CUresult);
        unsafe { unload(handle as *mut c_void) };
        return Err(1);
    }
    unsafe { *library = handle as *mut c_void };
    images::keep(handle, replacement.loaded_as(image, refused));
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryLoadFromFile(
    library: *mut *mut c_void,
    path: *const c_char,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> CUresult {
    let real = real!(cuLibraryLoadFromFile: LibraryLoad<*const c_char>);
    let from_file = || unsafe {
        real(
            library,
            path,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    let replacement = if path.is_null() {
        None
    } else {
        images::adapt_file("cuLibraryLoadFromFile", unsafe { CStr::from_ptr(path) })
    };
    let Some(replacement) = replacement else {
        let result = from_file();
        unsafe { images::loaded_as_is(result, library) };
        return result;
    };
    let loaded = unsafe {
        load_library(
            library,
            replacement,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    match loaded {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "cuLibraryLoadFromFile: the driver refused the adapted image (error {result}); loading the file"
            ));
            from_file()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryUnload(library: *mut c_void) -> CUresult {
    // Check if this is a lazy library: unload its sub-modules and stand-in.
    if let Some(result) = lazy::unload(library as usize) {
        fit::unload(library as usize);
        images::release(library as usize);
        return result;
    }
    let real = real!(cuLibraryUnload: unsafe extern "C" fn(*mut c_void) -> CUresult);
    let result = unsafe { real(library) };
    fit::unload(library as usize);
    images::release(library as usize);
    result
}

// ---- lazy kernel compilation hooks -------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetKernel(
    kernel: *mut *mut c_void,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    COUNTS[idx::cuLibraryGetKernel].fetch_add(1, Relaxed);
    let handle = library as usize;
    let find = || {
        if lazy::is_lazy(handle) {
            return unsafe { lazy::get_kernel(handle, name, kernel) };
        }
        let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult =
            unsafe { std::mem::transmute(REALS[idx::cuLibraryGetKernel].load(Relaxed)) };
        unsafe { real(kernel, library, name) }
    };
    let result = unsafe { lookup(handle, name, kernel, find) };
    if result == CUDA_SUCCESS && !kernel.is_null() && !name.is_null() {
        let kernel_handle = unsafe { *kernel } as usize;
        let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        fit::register_kernel(kernel_handle, handle, &name_str, true);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetKernelCount(count: *mut c_uint, library: *mut c_void) -> CUresult {
    COUNTS[idx::cuLibraryGetKernelCount].fetch_add(1, Relaxed);
    let handle = library as usize;
    if let Some(n) = lazy::kernel_count(handle) {
        if !count.is_null() {
            unsafe { *count = n as c_uint };
        }
        return CUDA_SUCCESS;
    }
    let real: unsafe extern "C" fn(*mut c_uint, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetKernelCount].load(Relaxed)) };
    unsafe { real(count, library) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryEnumerateKernels(
    kernels: *mut *mut c_void,
    count: c_uint,
    library: *mut c_void,
) -> CUresult {
    COUNTS[idx::cuLibraryEnumerateKernels].fetch_add(1, Relaxed);
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        return match unsafe { lazy::enumerate_kernels(handle, kernels, count) } {
            Ok(written) => {
                unsafe { register_enumerated_kernels(kernels, written, handle) };
                CUDA_SUCCESS
            }
            Err(code) => code,
        };
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, c_uint, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryEnumerateKernels].load(Relaxed)) };
    let result = unsafe { real(kernels, count, library) };
    if result == CUDA_SUCCESS {
        let written = unsafe { written_by_enumeration(library, count) };
        unsafe { register_enumerated_kernels(kernels, written, handle) };
    }
    result
}

/// How many handles the driver wrote when it enumerated the kernels of `library` (the handle
/// that the driver itself knows) into `capacity` places: the library's number of kernels, by
/// the driver's own count, or the capacity when that is smaller. The other places are the
/// program's memory as it was, which may hold anything: they are not read. Nothing when the
/// driver does not give the count.
///
/// # Safety
/// `library` must be a library that the driver loaded.
pub(crate) unsafe fn written_by_enumeration(library: *mut c_void, capacity: c_uint) -> usize {
    let count: unsafe extern "C" fn(*mut c_uint, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetKernelCount].load(Relaxed)) };
    let mut kernels: c_uint = 0;
    let result = unsafe { count(&mut kernels, library) };
    if result != CUDA_SUCCESS {
        log::info(format_args!(
            "cuLibraryEnumerateKernels: the driver does not give the library's number of kernels (error {result}): none of the handles is followed"
        ));
        return 0;
    }
    kernels.min(capacity) as usize
}

/// Register each kernel handle returned by `cuLibraryEnumerateKernels` with FIT, so that a launch
/// through one of them can be retried. The name of each kernel is obtained from the driver
/// (`cuKernelGetName`, with `cuFuncGetName` as a fallback: older drivers may not have the first).
///
/// # Safety
/// `kernels` must point to `written` kernel handles that the driver stored
/// (`written_by_enumeration`).
unsafe fn register_enumerated_kernels(kernels: *mut *mut c_void, written: usize, library: usize) {
    if kernels.is_null() {
        return;
    }
    let kernel_get_name: unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuKernelGetName].load(Relaxed)) };
    let func_get_name: unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuFuncGetName].load(Relaxed)) };
    for i in 0..written {
        let kernel = unsafe { *kernels.add(i) };
        if kernel.is_null() {
            continue;
        }
        let mut name_ptr: *const c_char = std::ptr::null();
        let got_name = unsafe { kernel_get_name(&mut name_ptr, kernel) } == CUDA_SUCCESS && !name_ptr.is_null();
        let got_name =
            got_name || (unsafe { func_get_name(&mut name_ptr, kernel) } == CUDA_SUCCESS && !name_ptr.is_null());
        if got_name {
            let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
            fit::register_kernel(kernel as usize, library, &name, true);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetModule(module: *mut *mut c_void, library: *mut c_void) -> CUresult {
    COUNTS[idx::cuLibraryGetModule].fetch_add(1, Relaxed);
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        // The stand-in has only the header; materialize the full module.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetModule].load(Relaxed)) };
                let result = unsafe { real(module, mat as *mut c_void) };
                unsafe { share_kernels(result, module, handle) };
                return result;
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetModule].load(Relaxed)) };
    let result = unsafe { real(module, library) };
    unsafe { share_kernels(result, module, handle) };
    result
}

/// The module the driver gave for a library has the kernels of that library: what is known of them
/// holds for lookups in the module too, and FIT needs the module to share the library's original
/// cubin so that a launch through `cuModuleGetFunction` on this module can be retried.
///
/// # Safety
/// `module` must be null or point to the handle the driver stored.
unsafe fn share_kernels(result: CUresult, module: *mut *mut c_void, library: usize) {
    if result == CUDA_SUCCESS && !module.is_null() {
        let module_handle = unsafe { *module } as usize;
        kernels::share(module_handle, library);
        fit::share_module(module_handle, library);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetGlobal(
    dptr: *mut u64,
    size: *mut usize,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    COUNTS[idx::cuLibraryGetGlobal].fetch_add(1, Relaxed);
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        // Globals live in the full module; materialize it.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetGlobal].load(Relaxed)) };
                let result = unsafe { real(dptr, size, mat as *mut c_void, name) };
                if result == CUDA_SUCCESS {
                    fit::asked_for_data(handle);
                }
                return result;
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetGlobal].load(Relaxed)) };
    let result = unsafe { real(dptr, size, library, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(handle);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetManaged(
    dptr: *mut u64,
    size: *mut usize,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    COUNTS[idx::cuLibraryGetManaged].fetch_add(1, Relaxed);
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        // Managed variables live in the full module; materialize it.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetManaged].load(Relaxed)) };
                let result = unsafe { real(dptr, size, mat as *mut c_void, name) };
                if result == CUDA_SUCCESS {
                    fit::asked_for_data(handle);
                }
                return result;
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetManaged].load(Relaxed)) };
    let result = unsafe { real(dptr, size, library, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(handle);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetUnifiedFunction(
    fptr: *mut *mut c_void,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    COUNTS[idx::cuLibraryGetUnifiedFunction].fetch_add(1, Relaxed);
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        // Unified functions live in the full module; materialize it.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetUnifiedFunction].load(Relaxed)) };
                return unsafe { real(fptr, mat as *mut c_void, name) };
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetUnifiedFunction].load(Relaxed)) };
    unsafe { real(fptr, library, name) }
}

/// `cuKernelGetLibrary`: for a kernel from a lazy library, return the lazy library handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelGetLibrary(library: *mut *mut c_void, kernel: *mut c_void) -> CUresult {
    COUNTS[idx::cuKernelGetLibrary].fetch_add(1, Relaxed);
    if let Some(lib_handle) = lazy::kernel_library(kernel as usize) {
        if !library.is_null() {
            unsafe { *library = lib_handle as *mut c_void };
        }
        return CUDA_SUCCESS;
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuKernelGetLibrary].load(Relaxed)) };
    unsafe { real(library, kernel) }
}

// ---- linker -------------------------------------------------------------------------------------------

type LinkAddData = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *mut c_void,
    usize,
    *const c_char,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
) -> CUresult;

/// Add `data` to a link, adapted if it is PTX or a fat binary. The linker reads its input during the
/// call, so the replacement does not outlive it.
unsafe fn link_add_data(
    who: &str,
    input: c_int,
    data: *mut c_void,
    size: usize,
    call: impl Fn(*mut c_void, usize) -> CUresult,
) -> CUresult {
    let replacement = match input {
        INPUT_PTX => unsafe { images::adapt(who, data, Some(size)) },
        INPUT_FATBINARY => unsafe { images::adapt(who, data, None) },
        _ => None,
    };
    let Some(replacement) = replacement else {
        return call(data, size);
    };
    if let Some(first) = replacement.refused().first() {
        // Linked code comes back as one image of machine code: its lookups cannot be told apart.
        log::warn(format_args!(
            "{who}: {} kernels of this code cannot run on this GPU (the first is {}: {}); nothing keeps them \
             from the program once linked, and launching one ends its CUDA context",
            replacement.refused().len(),
            first.kernel,
            first.reason
        ));
    }
    let result = call(replacement.as_ptr() as *mut c_void, replacement.len());
    if result == CUDA_SUCCESS {
        return result;
    }
    log::warn(format_args!(
        "{who}: the driver refused the adapted input (error {result}); adding the original"
    ));
    call(data, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkAddData(
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLinkAddData: LinkAddData);
    unsafe {
        link_add_data("cuLinkAddData", input, data, size, |data, size| {
            real(state, input, data, size, name, count, options, values)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkAddData_v2(
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLinkAddData_v2: LinkAddData);
    unsafe {
        link_add_data("cuLinkAddData", input, data, size, |data, size| {
            real(state, input, data, size, name, count, options, values)
        })
    }
}

// ---- failed launches (diagnostics) -------------------------------------------------------------------
//
// These two are only handed to the runtime when `VF_TRACE_LAUNCH=1` (see `TRACE_ONLY` in `lib.rs`);
// otherwise the runtime calls NVIDIA's functions directly.

/// `CUlaunchConfig`.
#[repr(C)]
pub struct LaunchConfig {
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub shared_memory: c_uint,
    pub stream: *mut c_void,
    pub attributes: *mut LaunchAttribute,
    pub attribute_count: c_uint,
}

/// `CUlaunchAttribute`: an identifier and a 64-byte value.
#[repr(C)]
pub struct LaunchAttribute {
    id: c_int,
    pad: [u8; 4],
    value: [u8; 64],
}

/// The name the driver gives for a kernel handle.
unsafe fn kernel_name(function: *mut c_void) -> String {
    let name_of = real!(cuFuncGetName: unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult);
    let mut name: *const c_char = std::ptr::null();
    if unsafe { name_of(&mut name, function) } == CUDA_SUCCESS && !name.is_null() {
        unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
    } else {
        "?".to_string()
    }
}

/// Launches per kernel name, kept when `VF_STATS` is set.
static LAUNCHES: std::sync::Mutex<Option<std::collections::HashMap<String, u64>>> = std::sync::Mutex::new(None);

/// Count one launch (`VF_STATS`). The name is asked from the driver at every launch: a handle may
/// name another kernel after its module was unloaded.
unsafe fn count_launch(function: *mut c_void) {
    let name = unsafe { kernel_name(function) };
    let mut launches = LAUNCHES.lock().unwrap_or_else(|e| e.into_inner());
    *launches.get_or_insert_with(Default::default).entry(name).or_default() += 1;
}

/// The launches counted so far by kernel name, most launched first.
pub(crate) fn launches() -> Option<Vec<(String, u64)>> {
    let launches = LAUNCHES.lock().unwrap_or_else(|e| e.into_inner());
    let mut sorted: Vec<(String, u64)> = launches.as_ref()?.iter().map(|(name, n)| (name.clone(), *n)).collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Some(sorted)
}

/// Log a launch the driver refused.
unsafe fn report_launch(
    function: *mut c_void,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared: c_uint,
    attributes: &[c_int],
    result: CUresult,
) {
    let name = unsafe { kernel_name(function) };
    log::error(format_args!(
        "launch of {name} failed with error {result}: grid {grid:?}, block {block:?}, shared memory {shared} bytes, \
         launch attributes {attributes:?}"
    ));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernel(
    function: *mut c_void,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLaunchKernel: unsafe extern "C" fn(
        *mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void,
    ) -> CUresult);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe {
        real(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_memory,
            stream,
            parameters,
            extra,
        )
    };
    // Refused for its registers (or so the driver says): from a second translation that keeps to
    // the kernel's own count, when there can be one (`fit`).
    if result == fit::OUT_OF_RESOURCES
        && let Some(fit_result) = unsafe {
            fit::retry(
                function,
                block_x.wrapping_mul(block_y).wrapping_mul(block_z),
                |second| {
                    real(
                        second,
                        grid_x,
                        grid_y,
                        grid_z,
                        block_x,
                        block_y,
                        block_z,
                        shared_memory,
                        stream,
                        parameters,
                        extra,
                    )
                },
            )
        }
    {
        if fit_result == CUDA_SUCCESS && config().profile {
            unsafe {
                profile::profile_launch(
                    function,
                    [grid_x, grid_y, grid_z],
                    [block_x, block_y, block_z],
                    shared_memory,
                    stream,
                )
            };
        }
        return fit_result;
    }
    if result != CUDA_SUCCESS && config().trace_launch {
        unsafe {
            report_launch(
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory,
                &[],
                result,
            )
        };
    }
    if result == CUDA_SUCCESS && config().profile {
        unsafe {
            profile::profile_launch(
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory,
                stream,
            )
        };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernelEx(
    launch: *const LaunchConfig,
    function: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLaunchKernelEx:
        unsafe extern "C" fn(*const LaunchConfig, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CUresult);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe { real(launch, function, parameters, extra) };
    // Refused for its registers: from a second translation, as in `cuLaunchKernel`.
    if result == fit::OUT_OF_RESOURCES
        && !launch.is_null()
        && let Some(fit_result) = unsafe {
            let block = (*launch).block;
            fit::retry(
                function,
                block[0].wrapping_mul(block[1]).wrapping_mul(block[2]),
                |second| real(launch, second, parameters, extra),
            )
        }
    {
        if fit_result == CUDA_SUCCESS && config().profile && !launch.is_null() {
            let launch_ref = unsafe { &*launch };
            unsafe {
                profile::profile_launch_ex(
                    function,
                    launch_ref.grid,
                    launch_ref.block,
                    launch_ref.shared_memory,
                    launch_ref.stream,
                )
            };
        }
        return fit_result;
    }
    if result != CUDA_SUCCESS && config().trace_launch && !launch.is_null() {
        let launch_ref = unsafe { &*launch };
        let attributes: Vec<c_int> = if launch_ref.attributes.is_null() {
            Vec::new()
        } else {
            (0..launch_ref.attribute_count as usize)
                .map(|i| unsafe { (*launch_ref.attributes.add(i)).id })
                .collect()
        };
        unsafe {
            report_launch(
                function,
                launch_ref.grid,
                launch_ref.block,
                launch_ref.shared_memory,
                &attributes,
                result,
            )
        };
    }
    if result == CUDA_SUCCESS && config().profile && !launch.is_null() {
        let launch_ref = unsafe { &*launch };
        unsafe {
            profile::profile_launch_ex(
                function,
                launch_ref.grid,
                launch_ref.block,
                launch_ref.shared_memory,
                launch_ref.stream,
            )
        };
    }
    result
}

// ---- per-thread default stream launch variants (`_ptsz`) -------------------------------------------
//
// The same logic as `cuLaunchKernel` and `cuLaunchKernelEx`, but each calls the driver's `_ptsz`
// entry so that the per-thread stream semantics of the program are preserved. The retry on 701
// must also go through the `_ptsz` entry of the driver.

type LaunchKernelFn = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

type LaunchKernelExFn =
    unsafe extern "C" fn(*const LaunchConfig, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernel_ptsz(
    function: *mut c_void,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLaunchKernel_ptsz: LaunchKernelFn);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe {
        real(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_memory,
            stream,
            parameters,
            extra,
        )
    };
    if result == fit::OUT_OF_RESOURCES
        && let Some(fit_result) = unsafe {
            fit::retry(
                function,
                block_x.wrapping_mul(block_y).wrapping_mul(block_z),
                |second| {
                    real(
                        second,
                        grid_x,
                        grid_y,
                        grid_z,
                        block_x,
                        block_y,
                        block_z,
                        shared_memory,
                        stream,
                        parameters,
                        extra,
                    )
                },
            )
        }
    {
        if fit_result == CUDA_SUCCESS && config().profile {
            unsafe {
                profile::profile_launch(
                    function,
                    [grid_x, grid_y, grid_z],
                    [block_x, block_y, block_z],
                    shared_memory,
                    stream,
                )
            };
        }
        return fit_result;
    }
    if result != CUDA_SUCCESS && config().trace_launch {
        unsafe {
            report_launch(
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory,
                &[],
                result,
            )
        };
    }
    if result == CUDA_SUCCESS && config().profile {
        unsafe {
            profile::profile_launch(
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory,
                stream,
            )
        };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernelEx_ptsz(
    launch: *const LaunchConfig,
    function: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLaunchKernelEx_ptsz: LaunchKernelExFn);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe { real(launch, function, parameters, extra) };
    if result == fit::OUT_OF_RESOURCES
        && !launch.is_null()
        && let Some(fit_result) = unsafe {
            let block = (*launch).block;
            fit::retry(
                function,
                block[0].wrapping_mul(block[1]).wrapping_mul(block[2]),
                |second| real(launch, second, parameters, extra),
            )
        }
    {
        if fit_result == CUDA_SUCCESS && config().profile && !launch.is_null() {
            let launch_ref = unsafe { &*launch };
            unsafe {
                profile::profile_launch_ex(
                    function,
                    launch_ref.grid,
                    launch_ref.block,
                    launch_ref.shared_memory,
                    launch_ref.stream,
                )
            };
        }
        return fit_result;
    }
    if result != CUDA_SUCCESS && config().trace_launch && !launch.is_null() {
        let launch_ref = unsafe { &*launch };
        let attributes: Vec<c_int> = if launch_ref.attributes.is_null() {
            Vec::new()
        } else {
            (0..launch_ref.attribute_count as usize)
                .map(|i| unsafe { (*launch_ref.attributes.add(i)).id })
                .collect()
        };
        unsafe {
            report_launch(
                function,
                launch_ref.grid,
                launch_ref.block,
                launch_ref.shared_memory,
                &attributes,
                result,
            )
        };
    }
    if result == CUDA_SUCCESS && config().profile && !launch.is_null() {
        let launch_ref = unsafe { &*launch };
        unsafe {
            profile::profile_launch_ex(
                function,
                launch_ref.grid,
                launch_ref.block,
                launch_ref.shared_memory,
                launch_ref.stream,
            )
        };
    }
    result
}
