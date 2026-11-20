//! Decides, for every export of a CUDA 13 library, what the forwarding library does with it.

use crate::elf::{Library, STT_FUNC};
use crate::libs::LibSpec;
use crate::protos::{AbiClass, Proto};
use std::collections::{BTreeMap, BTreeSet};

/// What a function returns, as far as a stub has to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Return {
    /// The library's status type: the stub returns its "not supported" value.
    Status,
    Void,
    Pointer,
    Float,
    Double,
    /// A basic integer type: 0.
    Int,
    /// No prototype in the headers, or a type the generator does not know. A library with a status type
    /// gets its "not supported" value (most undocumented entry points return a status); one without gets 0.
    Unknown,
}

impl Return {
    pub fn name(self) -> &'static str {
        match self {
            Return::Status => "status",
            Return::Void => "void",
            Return::Pointer => "pointer",
            Return::Float => "float",
            Return::Double => "double",
            Return::Int => "integer",
            Return::Unknown => "unknown",
        }
    }
}

pub fn classify_return(ret: Option<&str>, status_ty: Option<&str>) -> Return {
    let Some(ret) = ret else { return Return::Unknown };
    let ret = ret.trim();
    if Some(ret) == status_ty {
        return Return::Status;
    }
    match ret {
        "void" => Return::Void,
        "float" => Return::Float,
        "double" => Return::Double,
        "int"
        | "unsigned int"
        | "long"
        | "unsigned long"
        | "long long"
        | "long unsigned int"
        | "long long unsigned int"
        | "short"
        | "unsigned short"
        | "char"
        | "unsigned char"
        | "size_t"
        | "_Bool"
        | "ssize_t"
        | "int64_t"
        | "uint64_t"
        | "int32_t"
        | "uint32_t" => Return::Int,
        r if r.contains('*') => Return::Pointer,
        _ => Return::Unknown,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// A jump to `target` (the same name unless renamed) in the 12.9 library.
    Forward { target: String },
    /// A jump to `target`, and the status it returns is renumbered; `words` stack words are copied.
    ForwardStatus { target: String, words: usize },
    /// The 12.9 library has no such function, or has one that differs: a stub.
    Stub { ret: Return, reason: StubReason },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StubReason {
    Absent,
    /// Same name, different prototype: (12.9, 13.5).
    SignatureChanged(String, String),
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub version: Option<String>,
    pub action: Action,
    /// Why the function is not a plain forward, when it is adapted or renamed.
    pub adapted: Option<String>,
    /// Said once on standard error the first time the function is bound.
    pub note: Option<String>,
}

pub struct Inputs<'a> {
    pub spec: &'a LibSpec,
    pub new: &'a Library,
    pub old: &'a Library,
    pub new_protos: &'a BTreeMap<String, Proto>,
    pub old_protos: &'a BTreeMap<String, Proto>,
    /// The status enumerators changed value between the two versions: the status is renumbered.
    pub renumber: bool,
    /// `sizeof` of the by-value parameter types of the renumbered libraries (see `protos::type_sizes`).
    pub sizes: &'a BTreeMap<String, usize>,
    /// System V x86-64 ABI classification of the by-value parameter types (see `protos::type_abi_classes`).
    pub abi_classes: &'a BTreeMap<String, AbiClass>,
}

/// Exact number of 8-byte stack words that the System V x86-64 calling convention places on the
/// stack for this function's parameters.  Integers and pointers go in the first 6 integer registers
/// (rdi, rsi, rdx, rcx, r8, r9); floats and doubles go in the first 8 SSE registers (xmm0-7);
/// structures by value go in registers when they fit (classification from `abi_classes`) or on the
/// stack otherwise (MEMORY class, or not enough registers remain); the result is rounded up to an
/// even number for 16-byte stack alignment.
pub fn stack_words(
    proto: &Proto,
    sizes: &BTreeMap<String, usize>,
    abi_classes: &BTreeMap<String, AbiClass>,
) -> Result<usize, String> {
    let mut int_left: usize = 6; // rdi, rsi, rdx, rcx, r8, r9
    let mut sse_left: usize = 8; // xmm0 through xmm7
    let mut words: usize = 0;

    for param in crate::protos::split_params(&proto.params) {
        let param = param.trim().to_string();

        // Classify the parameter.
        match classify_param(&param, sizes, abi_classes)? {
            ParamClass::Integer => {
                if int_left > 0 {
                    int_left -= 1;
                } else {
                    words += 1;
                }
            }
            ParamClass::Sse => {
                if sse_left > 0 {
                    sse_left -= 1;
                } else {
                    words += 1;
                }
            }
            ParamClass::Struct {
                int_regs,
                sse_regs,
                stack_size,
            } => {
                if int_regs as usize <= int_left && sse_regs as usize <= sse_left {
                    int_left -= int_regs as usize;
                    sse_left -= sse_regs as usize;
                } else {
                    words += stack_size;
                }
            }
            ParamClass::Memory(stack_size) => {
                words += stack_size;
            }
        }
    }
    Ok(words + words % 2)
}

enum ParamClass {
    /// A scalar integer or pointer: occupies 1 integer register.
    Integer,
    /// A float or double: occupies 1 SSE register.
    Sse,
    /// A by-value struct that fits in registers (≤ 16 bytes, classified by the ABI probe).
    Struct {
        int_regs: u8,
        sse_regs: u8,
        stack_size: usize,
    },
    /// A by-value type that always goes on the stack (> 16 bytes, or classified as MEMORY).
    Memory(usize),
}

/// Classify a parameter type for the System V x86-64 calling convention.
fn classify_param(
    param: &str,
    sizes: &BTreeMap<String, usize>,
    abi_classes: &BTreeMap<String, AbiClass>,
) -> Result<ParamClass, String> {
    let param = param.trim();

    // Pointers, function pointers, arrays: always INTEGER, 1 register.
    if param.contains('*') || param.contains("(*") || param.contains('[') {
        return Ok(ParamClass::Integer);
    }

    // Check if it is a builtin scalar type.
    let words: Vec<&str> = param
        .split_whitespace()
        .filter(|w| !matches!(*w, "const" | "volatile" | "restrict"))
        .collect();
    let all_builtin = words.iter().all(|w| {
        matches!(
            *w,
            "int" | "unsigned" | "signed" | "long" | "short" | "char" | "float" | "double" | "_Bool"
        )
    });
    if all_builtin && !words.is_empty() {
        // float and double are SSE; long double is refused.
        if words.contains(&"float") && !words.contains(&"double") {
            return Ok(ParamClass::Sse);
        }
        if words.contains(&"double") {
            let longs = words.iter().filter(|w| **w == "long").count();
            if longs > 0 {
                // long double: on the stack, at a multiple of 16 bytes. The count here does not
                // follow alignments (a word of padding may come before it), so it is refused
                // and not miscounted; no function of the CUDA libraries takes one.
                return Err(format!(
                    "`{param}`: a long double on the stack is aligned to 16 bytes, which is not counted"
                ));
            }
            return Ok(ParamClass::Sse);
        }
        // Everything else is an integer type.
        return Ok(ParamClass::Integer);
    }

    // Named type (struct, enum, typedef): look up size and ABI class from the probes.
    let key = by_value_type(param);
    let size = sizes
        .get(&key)
        .copied()
        .ok_or_else(|| format!("size of `{param}` is not known"))?;

    if size > 16 {
        return Ok(ParamClass::Memory(size.div_ceil(8)));
    }

    let class = abi_classes
        .get(&key)
        .ok_or_else(|| format!("ABI classification of `{param}` is not known"))?;

    if class.int_regs == 0 && class.sse_regs == 0 {
        // MEMORY class despite size ≤ 16 (e.g. contains a long double member).
        return Ok(ParamClass::Memory(size.div_ceil(8)));
    }

    Ok(ParamClass::Struct {
        int_regs: class.int_regs,
        sse_regs: class.sse_regs,
        stack_size: size.div_ceil(8),
    })
}

/// The type of a parameter without qualifiers, as written in a `sizeof`.
pub fn by_value_type(param: &str) -> String {
    param
        .split_whitespace()
        .filter(|w| !matches!(*w, "const" | "volatile" | "restrict"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn plan(inp: &Inputs) -> Result<Vec<Entry>, String> {
    let status_ty = inp.spec.status.as_ref().map(|s| s.ty);
    let old_names: BTreeSet<&str> = inp.old.exports.iter().map(|s| s.name.as_str()).collect();
    let mut entries = Vec::new();
    for sym in &inp.new.exports {
        if sym.kind != STT_FUNC {
            return Err(format!(
                "{}: data or special symbol `{}` (type {}); not handled",
                inp.spec.new_name, sym.name, sym.kind
            ));
        }
        let name = sym.name.as_str();
        let p13 = inp.new_protos.get(name).filter(|p| !p.inline);
        let p12 = inp.old_protos.get(name).filter(|p| !p.inline);
        let mut entry = Entry {
            name: name.to_string(),
            version: sym.version.clone(),
            action: Action::Forward {
                target: name.to_string(),
            },
            adapted: None,
            note: None,
        };

        // Exact number of stack words the calling convention places on the stack for the
        // arguments of this function.  Undeclared functions that look like status functions get
        // the largest such count of any declared status function in the library.
        let max_words = inp
            .new_protos
            .values()
            .filter(|p| Some(p.ret.as_str()) == status_ty)
            .filter_map(|p| stack_words(p, inp.sizes, inp.abi_classes).ok())
            .max()
            .unwrap_or(0);
        let renumber_words = |declared: Option<&Proto>| -> Result<Option<usize>, String> {
            if !inp.renumber {
                return Ok(None);
            }
            match declared {
                Some(p) if Some(p.ret.as_str()) == status_ty => stack_words(p, inp.sizes, inp.abi_classes).map(Some),
                Some(_) => Ok(None),
                None if inp.spec.status_prefixes.iter().any(|pre| name.starts_with(pre)) => Ok(Some(max_words)),
                None => Ok(None),
            }
        };
        let forward = |target: String, declared: Option<&Proto>| -> Result<Action, String> {
            Ok(match renumber_words(declared)? {
                Some(words) => Action::ForwardStatus { target, words },
                None => Action::Forward { target },
            })
        };

        // Entry points that carry the library's version number.
        if let Some((from, to)) = inp.spec.rename_prefix
            && let Some(rest) = name.strip_prefix(from)
        {
            let target = format!("{to}{rest}");
            if old_names.contains(target.as_str()) {
                entry.action = forward(target.clone(), p13)?;
                entry.adapted = Some(format!(
                    "renamed: forwarded to `{target}` (the same entry point under 12.9's version number)"
                ));
            } else {
                entry.action = Action::Stub {
                    ret: classify_return(p13.map(|p| p.ret.as_str()), status_ty),
                    reason: StubReason::Absent,
                };
            }
            entries.push(entry);
            continue;
        }

        // A function whose 13 prototype is that of a versioned entry point of 12.9 (`__name_12_7`), while
        // 12.9's own `name` is a different, older function (cuFFT's JIT callback).
        if let Some(p13) = p13 {
            let alias = old_names.iter().find(|candidate| {
                candidate
                    .strip_prefix("__")
                    .and_then(|c| c.strip_prefix(name))
                    .is_some_and(|tail| {
                        tail.strip_prefix('_')
                            .is_some_and(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c == '_'))
                    })
            });
            let old_plain_differs =
                p12.is_none() || old_names.contains(name) && inp.old_protos.get(name).is_some_and(|p| p.inline);
            if let Some(alias) = alias
                && old_plain_differs
                && inp
                    .old_protos
                    .get(*alias)
                    .is_some_and(|a| a.ret == p13.ret && a.params == p13.params)
            {
                entry.action = forward(alias.to_string(), Some(p13))?;
                entry.adapted = Some(format!(
                    "12.9's `{name}` is an older function with another argument list; 13.5's prototype is that of `{alias}`, which has the same one"
                ));
                entries.push(entry);
                continue;
            }
        }

        // `...ForStream`: the 12.9 library has the heuristic or the check without the trailing stream.
        if inp.spec.for_stream
            && !old_names.contains(name)
            && let Some(base) = name.strip_suffix("ForStream")
            && old_names.contains(base)
        {
            let verified = match (p13, inp.new_protos.get(base)) {
                (Some(with), Some(without)) => {
                    let (a, b) = (
                        crate::protos::split_params(&with.params),
                        crate::protos::split_params(&without.params),
                    );
                    with.ret == without.ret
                        && a.len() == b.len() + 1
                        && a[..b.len()] == b[..]
                        && a[b.len()] == "cudaStream_t"
                }
                // The typed variants (`cublasLtACCMatmulAlgoGetHeuristicForStream`) are not in the
                // header: the name pattern stands, as for the declared pair.
                _ => true,
            };
            if verified {
                entry.action = forward(base.to_string(), p13)?;
                entry.adapted = Some(format!(
                    "forwarded to `{base}`: the trailing stream argument is ignored by 12.9"
                ));
                entry.note = Some(format!(
                    "{name}: the stream argument is ignored (CUDA 12.9's {base} takes none)"
                ));
                entries.push(entry);
                continue;
            }
        }

        if old_names.contains(name) {
            // The 12.9 headers define a wrapper of this name instead of declaring the exported function
            // (and no versioned entry point of 12.9 has 13.5's prototype): what 12.9 exports under the
            // name is not known to be what 13.5's header declares.
            if let (Some(a), Some(w)) = (p13, inp.old_protos.get(name))
                && w.inline
            {
                entry.action = Action::Stub {
                    ret: classify_return(Some(&a.ret), status_ty),
                    reason: StubReason::SignatureChanged(
                        format!("{} ({}) [inline wrapper in the 12.9 header]", w.ret, w.params),
                        format!("{} ({})", a.ret, a.params),
                    ),
                };
                entries.push(entry);
                continue;
            }
            match (p13, p12) {
                (Some(a), Some(b)) if (a.ret.as_str(), a.params.as_str()) != (b.ret.as_str(), b.params.as_str()) => {
                    entry.action = Action::Stub {
                        ret: classify_return(Some(&a.ret), status_ty),
                        reason: StubReason::SignatureChanged(
                            format!("{} ({})", b.ret, b.params),
                            format!("{} ({})", a.ret, a.params),
                        ),
                    };
                }
                _ => entry.action = forward(name.to_string(), p13)?,
            }
        } else {
            entry.action = Action::Stub {
                ret: classify_return(p13.map(|p| p.ret.as_str()), status_ty),
                reason: StubReason::Absent,
            };
        }
        entries.push(entry);
    }
    Ok(entries)
}

/// Functions that take the status type as an argument and are not adapted (a renumbered status would
/// reach 12.9 in 13.5's numbering).
pub fn status_arguments(inp: &Inputs, entries: &[Entry]) -> Vec<String> {
    let Some(status) = inp.spec.status.as_ref().filter(|_| inp.renumber) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter(|e| !matches!(e.action, Action::Stub { .. }))
        .filter(|e| {
            inp.new_protos.get(&e.name).is_some_and(|p| {
                crate::protos::split_params(&p.params)
                    .iter()
                    .any(|a| a.split(' ').any(|w| w == status.ty))
            })
        })
        .map(|e| e.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::{STT_FUNC, Symbol};
    use crate::libs::{LIBS, LibSpec};

    fn lib(names: &[&str]) -> Library {
        Library {
            exports: names
                .iter()
                .map(|n| Symbol {
                    name: n.to_string(),
                    version: Some("v".into()),
                    kind: STT_FUNC,
                    weak: false,
                })
                .collect(),
            ..Library::default()
        }
    }

    fn protos(list: &[(&str, &str, &str, bool)]) -> BTreeMap<String, Proto> {
        list.iter()
            .map(|(n, r, p, i)| {
                (
                    n.to_string(),
                    Proto {
                        ret: r.to_string(),
                        params: p.to_string(),
                        inline: *i,
                    },
                )
            })
            .collect()
    }

    fn spec(key: &str) -> &'static LibSpec {
        LIBS.iter().find(|l| l.key == key).unwrap()
    }

    #[test]
    fn classifies_returns() {
        assert_eq!(
            classify_return(Some("cublasStatus_t"), Some("cublasStatus_t")),
            Return::Status
        );
        assert_eq!(classify_return(Some("const char *"), None), Return::Pointer);
        assert_eq!(classify_return(Some("size_t"), None), Return::Int);
        assert_eq!(classify_return(None, None), Return::Unknown);
    }

    #[test]
    fn forwards_stubs_and_flags_changes() {
        let (new, old) = (lib(&["a", "b", "c"]), lib(&["a", "c"]));
        let p13 = protos(&[
            ("a", "cublasStatus_t", "int", false),
            ("b", "cublasStatus_t", "int", false),
            ("c", "cublasStatus_t", "int", false),
        ]);
        let p12 = protos(&[
            ("a", "cublasStatus_t", "int", false),
            ("c", "cublasStatus_t", "long", false),
        ]);
        let inp = Inputs {
            spec: spec("cublas"),
            new: &new,
            old: &old,
            new_protos: &p13,
            old_protos: &p12,
            renumber: false,
            sizes: &BTreeMap::new(),
            abi_classes: &BTreeMap::new(),
        };
        let e = plan(&inp).unwrap();
        assert_eq!(e[0].action, Action::Forward { target: "a".into() });
        assert_eq!(
            e[1].action,
            Action::Stub {
                ret: Return::Status,
                reason: StubReason::Absent
            }
        );
        assert!(matches!(
            e[2].action,
            Action::Stub {
                reason: StubReason::SignatureChanged(..),
                ..
            }
        ));
    }

    #[test]
    fn versioned_alias_and_for_stream() {
        let (new, old) = (
            lib(&["cufftXtSetJITCallback"]),
            lib(&["cufftXtSetJITCallback", "__cufftXtSetJITCallback_12_7"]),
        );
        let p13 = protos(&[("cufftXtSetJITCallback", "cufftResult", "int, void **", false)]);
        let p12 = protos(&[
            ("cufftXtSetJITCallback", "cufftResult", "int, void **", true),
            ("__cufftXtSetJITCallback_12_7", "cufftResult", "int, void **", false),
        ]);
        let inp = Inputs {
            spec: spec("cufft"),
            new: &new,
            old: &old,
            new_protos: &p13,
            old_protos: &p12,
            renumber: false,
            sizes: &BTreeMap::new(),
            abi_classes: &BTreeMap::new(),
        };
        assert_eq!(
            plan(&inp).unwrap()[0].action,
            Action::Forward {
                target: "__cufftXtSetJITCallback_12_7".into()
            }
        );

        let (new, old) = (lib(&["fooForStream", "foo"]), lib(&["foo"]));
        let p13 = protos(&[
            ("fooForStream", "cublasStatus_t", "int, cudaStream_t", false),
            ("foo", "cublasStatus_t", "int", false),
        ]);
        let inp = Inputs {
            spec: spec("cublasLt"),
            new: &new,
            old: &old,
            new_protos: &p13,
            old_protos: &p13,
            renumber: false,
            sizes: &BTreeMap::new(),
            abi_classes: &BTreeMap::new(),
        };
        assert_eq!(plan(&inp).unwrap()[0].action, Action::Forward { target: "foo".into() });
    }

    #[test]
    fn renumbered_status_copies_exact_stack_words() {
        // Three int params: all fit in registers (rdi, rsi, rdx), so 0 stack words.
        let (new, old) = (lib(&["nppiAdd"]), lib(&["nppiAdd"]));
        let p = protos(&[("nppiAdd", "NppStatus", "int, int, int", false)]);
        let inp = Inputs {
            spec: spec("nppial"),
            new: &new,
            old: &old,
            new_protos: &p,
            old_protos: &p,
            renumber: true,
            sizes: &BTreeMap::new(),
            abi_classes: &BTreeMap::new(),
        };
        assert_eq!(
            plan(&inp).unwrap()[0].action,
            Action::ForwardStatus {
                target: "nppiAdd".into(),
                words: 0
            }
        );
    }

    #[test]
    fn stack_words_with_integer_spill() {
        // 8 integer params: first 6 in registers, last 2 on the stack → 2 words.
        let sizes = BTreeMap::new();
        let abi = BTreeMap::new();
        let p = Proto {
            ret: "NppStatus".into(),
            params: "int, int, int, int, int, int, int, int".into(),
            inline: false,
        };
        assert_eq!(stack_words(&p, &sizes, &abi).unwrap(), 2);
    }

    #[test]
    fn stack_words_with_struct_on_stack() {
        // A large struct (> 16 bytes) always goes on the stack.
        let mut sizes = BTreeMap::new();
        sizes.insert("NppStreamContext".into(), 48usize);
        let mut abi = BTreeMap::new();
        abi.insert(
            "NppStreamContext".into(),
            AbiClass {
                int_regs: 0,
                sse_regs: 0,
            },
        );
        let p = Proto {
            ret: "NppStatus".into(),
            params: "int, int *, int, NppStreamContext".into(),
            inline: false,
        };
        // 3 int/ptr params in registers (rdi, rsi, rdx), NppStreamContext on stack: 48/8 = 6 words.
        assert_eq!(stack_words(&p, &sizes, &abi).unwrap(), 6);
    }

    #[test]
    fn stack_words_with_small_struct_in_regs() {
        // A small struct (8 bytes, INTEGER) fits in 1 int register.
        let mut sizes = BTreeMap::new();
        sizes.insert("NppiSize".into(), 8usize);
        let mut abi = BTreeMap::new();
        abi.insert(
            "NppiSize".into(),
            AbiClass {
                int_regs: 1,
                sse_regs: 0,
            },
        );
        let p = Proto {
            ret: "NppStatus".into(),
            params: "int *, int, int *, int, NppiSize".into(),
            inline: false,
        };
        // 5 params: all fit in 5 int registers, 0 on stack.
        assert_eq!(stack_words(&p, &sizes, &abi).unwrap(), 0);
    }

    #[test]
    fn stack_words_floats_past_eighth() {
        // 10 doubles: 8 in SSE registers, 2 on the stack → 2 words.
        let sizes = BTreeMap::new();
        let abi = BTreeMap::new();
        let p = Proto {
            ret: "NppStatus".into(),
            params: "double, double, double, double, double, double, double, double, double, double".into(),
            inline: false,
        };
        assert_eq!(stack_words(&p, &sizes, &abi).unwrap(), 2);
    }

    #[test]
    fn stack_words_with_a_struct_that_no_longer_fits() {
        // A structure of two integer eightbytes when one integer register is left: the whole
        // structure is on the stack, and the register goes to the next integer.
        let mut sizes = BTreeMap::new();
        sizes.insert("NppiRect".into(), 16usize);
        let mut abi = BTreeMap::new();
        abi.insert(
            "NppiRect".into(),
            AbiClass {
                int_regs: 2,
                sse_regs: 0,
            },
        );
        let p = Proto {
            ret: "NppStatus".into(),
            params: "int *, int, int *, int, int, NppiRect, int, int".into(),
            inline: false,
        };
        // Five in registers, the rectangle in two words, the next integer in the sixth register,
        // the last one in a word: three words, four with the stack kept at a multiple of 16 bytes.
        assert_eq!(stack_words(&p, &sizes, &abi).unwrap(), 4);
    }

    #[test]
    fn a_long_double_is_refused() {
        // On the stack at a multiple of 16 bytes: a word of padding may come before it, which
        // this count does not follow.
        let (sizes, abi) = (BTreeMap::new(), BTreeMap::new());
        let p = Proto {
            ret: "NppStatus".into(),
            params: "long, long, long, long, long, long, int, int, int, long double, long".into(),
            inline: false,
        };
        assert!(stack_words(&p, &sizes, &abi).is_err());
    }
}
