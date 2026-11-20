//! Finish: resolve branch targets and produce the final code bytes, once every
//! item has its word and its stall.

use std::collections::HashMap;

use crate::decode::Decoder;
use crate::field;

use super::util::{find_relative, put_operand_by_class};
use super::{Item, Refused};
use crate::decode::Decoded;

// ---- Finish ----

/// The code of a function, where each of its old offsets now is, and its register count.
pub(super) type Finished = (Vec<u8>, HashMap<u64, u64>, u32);

pub(super) fn finish(
    items: &mut [Item],
    src: &[Decoded],
    next_register: u32,
    dec70: &mut Decoder,
) -> Result<Finished, Refused> {
    // Decode any items that need it (inserted NOPs)
    for item in items.iter_mut() {
        if item.d.is_none() {
            item.d = Some(
                dec70
                    .decode(item.word)
                    .ok_or_else(|| Refused("inserted NOP is not a valid sm_70 instruction".into()))?,
            );
        }
    }

    let mut new_of: HashMap<u64, u64> = HashMap::new();
    for (k, item) in items.iter().enumerate() {
        if item.first {
            new_of.insert(16 * item.origin as u64, 16 * k as u64);
        }
    }
    new_of.insert(16 * src.len() as u64, 16 * items.len() as u64);

    for (k, item) in items.iter_mut().enumerate() {
        let Some(d) = item.d.clone() else { continue };
        if let Some(target) = item.target {
            let Some((rel_fid, scale)) = find_relative(&d, 70) else {
                return Err(Refused(format!(
                    "the translation of the branch at 0x{:x} has nowhere to say where it goes",
                    16 * item.origin
                )));
            };
            let new_target = new_of[&target];
            let new_offset = (new_target as i64 - 16 * k as i64 - 16).div_euclid(scale as i64);
            let mask = (1u64 << field::width(rel_fid)) - 1;
            item.word = field::put(rel_fid, item.word, (new_offset as u64) & mask);
        }
        if let Some(fix) = item.fix {
            item.word = put_operand_by_class(d.name, item.word, "Sb", new_of[&fix], false)?;
        }
        if dec70.decode(item.word).is_none() {
            return Err(Refused(format!(
                "after its operands were set, the instruction for 0x{:x} is not an sm_70 instruction",
                16 * item.origin
            )));
        }
    }

    let code: Vec<u8> = items
        .iter()
        .flat_map(|item| item.word.to_le_bytes().into_iter().take(16))
        .collect();
    let count = next_register + 2;

    Ok((code, new_of, count))
}
