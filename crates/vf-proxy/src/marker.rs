//! What a program that knows of the layer can ask it.
//!
//! A program built for later GPUs is told the compute capability that its code starts at (7.5,
//! `callers`), so that it loads that code; the name of the card is still the V100's. What it
//! cannot tell from the driver's answers is that the card is a Volta with the layer in front
//! of it: that its code will run translated, with what Volta lacks emulated. A program that
//! has kernels written for Volta beside its code for later GPUs may then prefer them (PTX for
//! `sm_70`, which the driver compiles as it is).
//!
//! These functions are in `libcuda.so.1` only when it is the layer's: `dlsym` finds them, or
//! does not. They take no argument, call nothing of the driver and may be called at any time.
//!
//! - `volta_forever_abi()`: the number of this interface, 1. A later interface keeps what an
//!   earlier one has.
//! - `volta_forever_version()`: the version of the layer, as text that lives as long as the
//!   library.

use std::ffi::{c_char, c_int};

/// The number of the interface of this module.
pub const ABI: c_int = 1;

/// The layer's version, with the zero that ends a C string.
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// The number of this interface: 1.
#[unsafe(no_mangle)]
pub extern "C" fn volta_forever_abi() -> c_int {
    ABI
}

/// The version of the layer.
#[unsafe(no_mangle)]
pub extern "C" fn volta_forever_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn the_version_is_a_c_string_of_the_workspace_version() {
        let version = unsafe { CStr::from_ptr(volta_forever_version()) };
        assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
        assert_eq!(volta_forever_abi(), 1);
    }
}
