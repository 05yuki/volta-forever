//! Field operations on 128-bit instruction words.
//!
//! A field is a set of bit positions within a 128-bit instruction word. The generator
//! stores each field as a slice of FIELD_BITS (bit positions, most significant first) and
//! a precomputed mask. Most fields are contiguous bit ranges; for those, the `start_bit`
//! in the FIELDS table enables a fast path.
//!
//! Values are u64 because some fields (branch offsets) are up to 48 bits wide.

use crate::generated::fields::{FIELD_BITS, FIELDS};

/// Get the value of field `fid` from a 128-bit instruction word.
/// Returns u64 because fields can be up to 48 bits (branch offsets).
#[inline]
pub fn get(fid: u16, word: u128) -> u64 {
    let (offset, width, _mask_lo, _mask_hi, start) = FIELDS[fid as usize];
    if start != 255 {
        ((word >> start) & ((1u128 << width) - 1)) as u64
    } else {
        let bits = &FIELD_BITS[offset as usize..(offset as usize + width as usize)];
        let mut value = 0u64;
        for &b in bits {
            value = (value << 1) | ((word >> b) as u64 & 1);
        }
        value
    }
}

/// Set the value of field `fid` in a 128-bit instruction word.
#[inline]
pub fn put(fid: u16, word: u128, value: u64) -> u128 {
    let (offset, width, mask_lo, mask_hi, start) = FIELDS[fid as usize];
    let mask = (mask_hi as u128) << 64 | mask_lo as u128;
    if start != 255 {
        (word & !mask) | (((value as u128) << start) & mask)
    } else {
        let bits = &FIELD_BITS[offset as usize..(offset as usize + width as usize)];
        let mut w = word;
        for (i, &b) in bits.iter().rev().enumerate() {
            w = (w & !(1u128 << b)) | (((value as u128 >> i) & 1) << b);
        }
        w
    }
}

/// The number of bits in a field.
#[inline]
pub fn width(fid: u16) -> u8 {
    FIELDS[fid as usize].1
}

/// Sign-extend the value of field `fid` from the instruction word.
/// Returns i64 because fields can be up to 48 bits.
#[inline]
pub fn signed(fid: u16, word: u128) -> i64 {
    let v = get(fid, word);
    let w = width(fid);
    if w > 0 && w < 64 && (v >> (w - 1)) & 1 != 0 {
        v as i64 - (1i64 << w)
    } else {
        v as i64
    }
}
