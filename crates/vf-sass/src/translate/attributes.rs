//! Attribute helpers: check, rewrite and strip the `.nv.info.*` attribute sections
//! when the offsets in the code change with the translation.

use std::collections::HashMap;

use super::{OFFSET_LISTS, OFFSET_PAIRS, PLAIN_ATTRS, Refused};

pub(super) fn check_attributes(data: &[u8], new_of: &HashMap<u64, u64>) -> Result<(), Refused> {
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let fmt = data[pos];
        let attr = data[pos + 1];
        let val = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()) as usize;
        let size = if fmt == 4 { 4 + val } else { 4 };
        if fmt == 4 {
            let places = attr_places(data, attr, pos, size)?;
            for at in places {
                if at + 4 <= data.len() {
                    let old = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as u64;
                    if !new_of.contains_key(&old) {
                        return Err(Refused(format!(
                            "attribute 0x{:02x} points at 0x{:x}, which is not an instruction",
                            attr, old
                        )));
                    }
                }
            }
        }
        pos += size;
    }
    Ok(())
}

pub(super) fn rewrite_attributes(data: &[u8], new_of: &HashMap<u64, u64>) -> Vec<u8> {
    let mut out = data.to_vec();
    let mut pos = 0;
    while pos + 4 <= out.len() {
        let fmt = out[pos];
        let attr = out[pos + 1];
        let val = u16::from_le_bytes(out[pos + 2..pos + 4].try_into().unwrap()) as usize;
        let size = if fmt == 4 { 4 + val } else { 4 };
        if fmt == 4
            && let Ok(places) = attr_places(&out, attr, pos, size)
        {
            for at in places {
                if at + 4 <= out.len() {
                    let old = u32::from_le_bytes(out[at..at + 4].try_into().unwrap()) as u64;
                    if let Some(&new_off) = new_of.get(&old) {
                        out[at..at + 4].copy_from_slice(&(new_off as u32).to_le_bytes());
                    }
                }
            }
        }
        pos += size;
    }
    out
}

fn attr_places(data: &[u8], attr: u8, pos: usize, size: usize) -> Result<Vec<usize>, Refused> {
    if OFFSET_LISTS.contains(&attr) {
        Ok((pos + 4..pos + size).step_by(4).collect())
    } else if let Some(&(_, p)) = OFFSET_PAIRS.iter().find(|&&(a, _)| a == attr) {
        Ok((pos + 4 + 4 * p as usize..pos + size).step_by(8).collect())
    } else if PLAIN_ATTRS.contains(&attr) {
        Ok(Vec::new())
    } else if attr == 0x34 {
        // Jump tables (`tables_of`): the branch and the places it leads to.
        let mut places = Vec::new();
        let mut at = pos + 4;
        while at + 12 <= pos + size {
            let count = if at + 12 <= data.len() {
                u32::from_le_bytes(data[at + 8..at + 12].try_into().unwrap()) as usize
            } else {
                0
            };
            if at + 12 + 4 * count > pos + size {
                return Err(Refused("a record of jump tables that is not understood".into()));
            }
            places.push(at);
            places.extend((at + 12..at + 12 + 4 * count).step_by(4));
            at += 12 + 4 * count;
        }
        Ok(places)
    } else {
        Err(Refused(format!("attribute 0x{:02x}, which is not known", attr)))
    }
}

pub(super) fn find_exits(data: &[u8]) -> Vec<u32> {
    let mut exits = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let fmt = data[pos];
        let attr = data[pos + 1];
        let val = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()) as usize;
        if fmt == 4 && attr == 0x1c {
            // The last such attribute counts.
            exits.clear();
            let mut at = pos + 4;
            while at + 4 <= (pos + 4 + val).min(data.len()) {
                exits.push(u32::from_le_bytes(data[at..at + 4].try_into().unwrap()));
                at += 4;
            }
        }
        pos += if fmt == 4 { 4 + val } else { 4 };
    }
    exits
}

pub(super) fn strip_attributes(data: &[u8]) -> Vec<u8> {
    let mut drop_set: Vec<u8> = OFFSET_LISTS.iter().filter(|&&a| a != 0x1c).copied().collect();
    drop_set.extend(OFFSET_PAIRS.iter().map(|&(a, _)| a));
    drop_set.push(0x34);
    strip_attrs_by_kind(data, &drop_set)
}

pub(super) fn strip_attrs_by_kind(data: &[u8], drop: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let fmt = data[pos];
        let attr = data[pos + 1];
        let val = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()) as usize;
        let size = if fmt == 4 { 4 + val } else { 4 };
        if !drop.contains(&attr) {
            out.extend_from_slice(&data[pos..(pos + size).min(data.len())]);
        }
        pos += size;
    }
    out
}
