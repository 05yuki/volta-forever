//! Reads what the generator needs from a shared library: the dynamic symbols with their symbol
//! versions, the SONAME, the NEEDED list, and the version definitions.

use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    /// Name of the version node the symbol is bound to, if the library has versions.
    pub version: Option<String>,
    pub kind: u8,
    pub weak: bool,
}

pub const STT_FUNC: u8 = 2;

#[derive(Debug, Default)]
pub struct Library {
    pub soname: Option<String>,
    pub needed: Vec<String>,
    /// Defined symbols (the version pseudo-symbols excluded), sorted by name.
    pub exports: Vec<Symbol>,
    /// Undefined symbols, sorted by name.
    pub imports: Vec<Symbol>,
    /// Version definitions: (index, name, number of names, so parents + 1), in file order.
    pub verdefs: Vec<(u16, String, u16)>,
}

fn u16_at(b: &[u8], o: usize) -> Result<u16, String> {
    b.get(o..o + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| "truncated ELF".to_string())
}
fn u32_at(b: &[u8], o: usize) -> Result<u32, String> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| "truncated ELF".to_string())
}
fn u64_at(b: &[u8], o: usize) -> Result<u64, String> {
    b.get(o..o + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| "truncated ELF".to_string())
}
fn cstr(b: &[u8], o: usize) -> Result<String, String> {
    let s = b.get(o..).ok_or("string out of range")?;
    let n = s.iter().position(|&c| c == 0).ok_or("unterminated string")?;
    Ok(String::from_utf8_lossy(&s[..n]).into_owned())
}

struct Section {
    kind: u32,
    offset: usize,
    size: usize,
    link: usize,
}

pub fn read(path: &Path) -> Result<Library, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse(b: &[u8]) -> Result<Library, String> {
    if b.get(..4) != Some(b"\x7fELF") || b.get(4) != Some(&2) || b.get(5) != Some(&1) {
        return Err("not a little-endian ELF64 file".into());
    }
    let shoff = u64_at(b, 0x28)? as usize;
    let shentsize = u16_at(b, 0x3a)? as usize;
    let shnum = u16_at(b, 0x3c)? as usize;
    let mut sections = Vec::new();
    for i in 0..shnum {
        let o = shoff + i * shentsize;
        sections.push(Section {
            kind: u32_at(b, o + 4)?,
            offset: u64_at(b, o + 0x18)? as usize,
            size: u64_at(b, o + 0x20)? as usize,
            link: u32_at(b, o + 0x28)? as usize,
        });
    }
    let find = |kind: u32| sections.iter().find(|s| s.kind == kind);
    let mut lib = Library::default();

    if let Some(dynamic) = find(6) {
        let strtab = &sections[dynamic.link];
        for k in 0..dynamic.size / 16 {
            let tag = u64_at(b, dynamic.offset + k * 16)?;
            let val = u64_at(b, dynamic.offset + k * 16 + 8)? as usize;
            match tag {
                0 => break,
                1 => lib.needed.push(cstr(b, strtab.offset + val)?),
                14 => lib.soname = Some(cstr(b, strtab.offset + val)?),
                _ => {}
            }
        }
    }

    let mut version_names: BTreeMap<u16, String> = BTreeMap::new();
    if let Some(verdef) = find(0x6fff_fffd) {
        let strtab = &sections[verdef.link];
        let mut o = verdef.offset;
        loop {
            let flags = u16_at(b, o + 2)?;
            let ndx = u16_at(b, o + 4)?;
            let cnt = u16_at(b, o + 6)?;
            let aux = u32_at(b, o + 12)? as usize;
            let next = u32_at(b, o + 16)? as usize;
            let name = cstr(b, strtab.offset + u32_at(b, o + aux)? as usize)?;
            // Flag 1 is the BASE entry, which carries the file's own name.
            if flags & 1 == 0 {
                lib.verdefs.push((ndx, name.clone(), cnt));
            }
            version_names.insert(ndx, name);
            if next == 0 {
                break;
            }
            o += next;
        }
    }
    let mut needed_versions: BTreeMap<u16, String> = BTreeMap::new();
    if let Some(verneed) = find(0x6fff_fffe) {
        let strtab = &sections[verneed.link];
        let mut o = verneed.offset;
        loop {
            let cnt = u16_at(b, o + 2)?;
            let aux = u32_at(b, o + 8)? as usize;
            let next = u32_at(b, o + 12)? as usize;
            let mut a = o + aux;
            for _ in 0..cnt {
                let other = u16_at(b, a + 6)?;
                let name = cstr(b, strtab.offset + u32_at(b, a + 8)? as usize)?;
                needed_versions.insert(other, name);
                let anext = u32_at(b, a + 12)? as usize;
                if anext == 0 {
                    break;
                }
                a += anext;
            }
            if next == 0 {
                break;
            }
            o += next;
        }
    }

    let dynsym = find(11).ok_or("no .dynsym")?;
    let strtab = &sections[dynsym.link];
    let versym = find(0x6fff_ffff);
    for i in 1..dynsym.size / 24 {
        let o = dynsym.offset + i * 24;
        let name = cstr(b, strtab.offset + u32_at(b, o)? as usize)?;
        let info = b[o + 4];
        let shndx = u16_at(b, o + 6)?;
        let (kind, bind) = (info & 15, info >> 4);
        if shndx == 0xfff1 {
            continue; // the version nodes themselves
        }
        let ver_index = match versym {
            Some(v) => u16_at(b, v.offset + i * 2)?,
            None => 0,
        };
        let id = ver_index & 0x7fff;
        let defined = shndx != 0;
        let version = if id < 2 {
            None
        } else if defined {
            version_names.get(&id).cloned()
        } else {
            needed_versions.get(&id).cloned()
        };
        let sym = Symbol {
            name,
            version,
            kind,
            weak: bind == 2,
        };
        if defined {
            lib.exports.push(sym)
        } else {
            lib.imports.push(sym)
        }
    }
    lib.exports.sort_by(|a, b| a.name.cmp(&b.name));
    lib.imports.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(lib)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_system_libc() {
        let lib = read(Path::new("/lib/x86_64-linux-gnu/libc.so.6")).unwrap();
        assert_eq!(lib.soname.as_deref(), Some("libc.so.6"));
        let malloc = lib.exports.iter().find(|s| s.name == "malloc").unwrap();
        assert_eq!(malloc.kind, STT_FUNC);
        assert!(malloc.version.as_deref().unwrap().starts_with("GLIBC_"));
    }

    #[test]
    fn rejects_other_files() {
        assert!(parse(b"not an elf file").is_err());
    }
}
