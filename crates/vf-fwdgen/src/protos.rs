//! Function prototypes of a toolkit's headers, as the C compiler reads them (`gcc -aux-info`: one
//! normalised line per declaration, macros expanded, so names are the link-level names), and the
//! enumerators of the status types.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proto {
    pub ret: String,
    /// The parameter types, comma separated as the compiler prints them (`void` when there are none).
    pub params: String,
    /// A function defined in the header (static inline), not a declaration of an exported function.
    pub inline: bool,
}

/// Splits a parameter list at the commas outside parentheses.
pub fn split_params(params: &str) -> Vec<String> {
    let params = params.trim();
    if params.is_empty() || params == "void" {
        return Vec::new();
    }
    let (mut depth, mut start, mut out) = (0i32, 0, Vec::new());
    for (i, c) in params.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(params[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(params[start..].trim().to_string());
    out
}

fn gcc(include: &Path) -> Command {
    let mut c = Command::new("gcc");
    c.arg("-w").arg(format!("-I{}", include.display()));
    c
}

/// Prototypes declared by the headers of one group, compiled together as one C file.
pub fn read_group(
    include: &Path,
    headers: &[&str],
    defines: &[&str],
    work: &Path,
    tag: &str,
) -> Result<BTreeMap<String, Proto>, String> {
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let source = work.join(format!("{tag}.c"));
    let aux = work.join(format!("{tag}.aux"));
    let text: String = headers.iter().map(|h| format!("#include <{h}>\n")).collect();
    std::fs::write(&source, text).map_err(|e| e.to_string())?;
    let out = gcc(include)
        .args(defines.iter().map(|d| format!("-D{d}")))
        .args(["-fsyntax-only", "-aux-info"])
        .arg(&aux)
        .arg(&source)
        .output()
        .map_err(|e| format!("gcc: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "gcc on {headers:?} in {}: {}",
            include.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let aux = std::fs::read_to_string(&aux).map_err(|e| e.to_string())?;
    Ok(parse_aux_info(&aux))
}

pub fn parse_aux_info(text: &str) -> BTreeMap<String, Proto> {
    let mut map = BTreeMap::new();
    for line in text.lines() {
        // /* file:line:NC */ extern int name (params);      (F instead of C for a definition)
        let Some(rest) = line.strip_prefix("/*") else { continue };
        let Some((tag, decl)) = rest.split_once("*/") else {
            continue;
        };
        let inline = tag.trim_end().ends_with("F");
        let decl = decl.trim().strip_suffix(';').unwrap_or(decl.trim());
        let decl = decl
            .strip_prefix("extern ")
            .or_else(|| decl.strip_prefix("static "))
            .unwrap_or(decl);
        let Some(open) = decl.find(" (") else { continue };
        let Some(params) = decl[open + 2..].strip_suffix(')') else {
            continue;
        };
        let head = decl[..open].trim_end();
        let name_start = head
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        let (ret, name) = (head[..name_start].trim().to_string(), head[name_start..].to_string());
        if name.is_empty() || ret.is_empty() {
            continue;
        }
        map.entry(name).or_insert(Proto {
            ret,
            params: params.trim().to_string(),
            inline,
        });
    }
    map
}

/// Size in bytes of a parameter type that is not a pointer, when it can be told from the spelling.
pub fn builtin_size(ty: &str) -> Option<usize> {
    let ty = ty.trim();
    if ty.contains('*') || ty.contains("(*") || ty.contains('[') {
        return Some(8);
    }
    let words: Vec<&str> = ty
        .split_whitespace()
        .filter(|w| !matches!(*w, "const" | "volatile" | "restrict"))
        .collect();
    let all_builtin = words.iter().all(|w| {
        matches!(
            *w,
            "int" | "unsigned" | "signed" | "long" | "short" | "char" | "float" | "double" | "_Bool"
        )
    });
    if !all_builtin {
        return None;
    }
    let longs = words.iter().filter(|w| **w == "long").count();
    Some(if words.contains(&"double") {
        if longs > 0 { 16 } else { 8 }
    } else if longs > 0 {
        8
    } else if words.contains(&"short") {
        2
    } else if words.contains(&"char") || words.contains(&"_Bool") {
        1
    } else {
        4
    })
}

/// ABI classification of a named type for the System V x86-64 calling convention: how many integer
/// and SSE registers it consumes when passed by value as a function argument.  When both are 0 the
/// type has MEMORY class (always passed on the stack, size > 16 bytes or the compiler classifies it
/// that way).  Determined by compiling probe functions with `gcc -O2 -S` and inspecting the generated
/// assembly, so the classification comes from the compiler, not from a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbiClass {
    pub int_regs: u8,
    pub sse_regs: u8,
}

/// For each named type, determine its System V x86-64 ABI classification by compiling two probe
/// functions per type to assembly and inspecting which registers carry the second argument.
///
///  *  `checkA(T x, long m) -> long { return m; }`  —  `m` lands in the integer register after the
///     ones `T` consumed.  Inspecting which register holds the return value tells us `int_regs`.
///  *  `checkB(T x, double d) -> long { memcpy(&r, &d, 8); return r; }`  —  similarly for SSE.
pub fn type_abi_classes(
    include: &Path,
    headers: &[&str],
    types: &[(String, usize)],
    work: &Path,
    tag: &str,
) -> Result<BTreeMap<String, AbiClass>, String> {
    let mut classes = BTreeMap::new();
    if types.is_empty() {
        return Ok(classes);
    }
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;

    let mut code: String = headers.iter().map(|h| format!("#include <{h}>\n")).collect();
    code.push_str("#include <string.h>\n");

    // Mangle type names into valid C identifiers.
    let mangle = |t: &str| -> String {
        t.chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
            .collect()
    };

    for (ty, size) in types {
        // Types larger than 16 bytes are always MEMORY (ABI rule): no probe needed.
        if *size > 16 {
            classes.insert(
                ty.clone(),
                AbiClass {
                    int_regs: 0,
                    sse_regs: 0,
                },
            );
            continue;
        }
        let m = mangle(ty);
        // Probe A: how many integer registers does T consume?
        // After T, the next long argument `m` lands in the first free integer register.
        // The function just returns m, so the assembly is `movq %REG, %rax; ret`.
        //   %rdi → T consumed 0 int regs     %rsi → 1     %rdx → 2
        code.push_str(&format!(
            "__attribute__((noipa)) long vf_abi_a_{m}({ty} x, long m) {{ (void)x; return m; }}\n"
        ));
        // Probe B: how many SSE registers does T consume?
        //   %xmm0 → 0     %xmm1 → 1     %xmm2 → 2
        code.push_str(&format!(
            "__attribute__((noipa)) long vf_abi_b_{m}({ty} x, double d) {{ (void)x; long r; __builtin_memcpy(&r, &d, 8); return r; }}\n"
        ));
    }

    let source = work.join(format!("{tag}.abi.c"));
    let asm_path = work.join(format!("{tag}.abi.s"));
    std::fs::write(&source, &code).map_err(|e| e.to_string())?;
    let out = gcc(include)
        .args(["-O2", "-S", "-o"])
        .arg(&asm_path)
        .arg(&source)
        .output()
        .map_err(|e| format!("gcc: {e}"))?;
    if !out.status.success() {
        return Err(format!("ABI probe: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let asm = std::fs::read_to_string(&asm_path).map_err(|e| e.to_string())?;

    for (ty, size) in types {
        if *size > 16 {
            continue; // already classified
        }
        let m = mangle(ty);
        let int_regs = abi_probe_int_regs(&asm, &format!("vf_abi_a_{m}"))
            .map_err(|e| format!("ABI classification of `{ty}` (int regs): {e}"))?;
        let sse_regs = abi_probe_sse_regs(&asm, &format!("vf_abi_b_{m}"))
            .map_err(|e| format!("ABI classification of `{ty}` (SSE regs): {e}"))?;
        if int_regs == 0 && sse_regs == 0 {
            // Size ≤ 16 but classified as MEMORY (e.g. contains long double or x87 member).
            // Fine: it goes on the stack.
        }
        classes.insert(ty.clone(), AbiClass { int_regs, sse_regs });
    }
    Ok(classes)
}

/// Parse the assembly of a probe-A function (`return m` where m follows T) to find how many
/// integer registers T consumed.  The function body is `movq %REG, %rax; ret`:
///   %rdi → 0,  %rsi → 1,  %rdx → 2.
fn abi_probe_int_regs(asm: &str, name: &str) -> Result<u8, String> {
    let body = extract_function_body(asm, name)?;
    // Look for a mov from an integer register to %rax / %eax.
    for line in body.lines() {
        let t = line.trim();
        if !t.starts_with("mov") {
            continue;
        }
        if (t.contains("%rax") || t.contains("%eax")) && t.contains(", ") {
            // AT&T: source, dest — check the source (before the comma).
            let source = t.split(',').next().unwrap_or("");
            if source.contains("%rdi") || source.contains("%edi") {
                return Ok(0);
            }
            if source.contains("%rsi") || source.contains("%esi") {
                return Ok(1);
            }
            if source.contains("%rdx") || source.contains("%edx") {
                return Ok(2);
            }
        }
    }
    Err(format!(
        "cannot determine integer register usage from assembly of {name}"
    ))
}

/// Parse the assembly of a probe-B function (`return *(long*)&d` where d follows T) to find how
/// many SSE registers T consumed:  %xmm0 → 0,  %xmm1 → 1,  %xmm2 → 2.
fn abi_probe_sse_regs(asm: &str, name: &str) -> Result<u8, String> {
    let body = extract_function_body(asm, name)?;
    for line in body.lines() {
        let t = line.trim();
        if !t.starts_with("mov") {
            continue;
        }
        if (t.contains("%rax") || t.contains("%eax")) && t.contains(", ") {
            let source = t.split(',').next().unwrap_or("");
            if source.contains("%xmm0") {
                return Ok(0);
            }
            if source.contains("%xmm1") {
                return Ok(1);
            }
            if source.contains("%xmm2") {
                return Ok(2);
            }
        }
    }
    Err(format!("cannot determine SSE register usage from assembly of {name}"))
}

/// Extract the instructions of a function from GCC's assembly output (between the label and ret).
fn extract_function_body(asm: &str, name: &str) -> Result<String, String> {
    let label = format!("{name}:");
    let start = asm
        .find(&label)
        .ok_or_else(|| format!("function `{name}` not found in assembly output"))?;
    let body = &asm[start + label.len()..];
    // End at the first `ret` instruction.
    let end = body
        .find("\tret")
        .ok_or_else(|| format!("no `ret` found for `{name}`"))?;
    Ok(body[..end + 4].to_string())
}

/// `sizeof` of each named type, by compiling and running a small program against the headers.
pub fn type_sizes(
    include: &Path,
    headers: &[&str],
    types: &[String],
    work: &Path,
    tag: &str,
) -> Result<BTreeMap<String, usize>, String> {
    let mut sizes = BTreeMap::new();
    if types.is_empty() {
        return Ok(sizes);
    }
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let mut code: String = headers.iter().map(|h| format!("#include <{h}>\n")).collect();
    code.push_str("#include <stdio.h>\nint main(void) {\n");
    for t in types {
        code.push_str(&format!("    printf(\"{t}|%zu\\n\", sizeof({t}));\n"));
    }
    code.push_str("    return 0;\n}\n");
    let source = work.join(format!("{tag}.sizes.c"));
    std::fs::write(&source, code).map_err(|e| e.to_string())?;
    let exe = work.join(format!("{tag}.sizes"));
    let built = gcc(include)
        .arg("-o")
        .arg(&exe)
        .arg(&source)
        .output()
        .map_err(|e| format!("gcc: {e}"))?;
    if !built.status.success() {
        return Err(format!("sizeof probe: {}", String::from_utf8_lossy(&built.stderr)));
    }
    let run = Command::new(&exe)
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    for line in String::from_utf8_lossy(&run.stdout).lines() {
        let (t, n) = line.split_once('|').ok_or("bad probe line")?;
        sizes.insert(t.to_string(), n.parse::<usize>().map_err(|e| e.to_string())?);
    }
    Ok(sizes)
}

/// Enumerators (name, value) of the enum type `typedef_name`, read from the headers and evaluated by
/// compiling and running a small program on the host.
pub fn enum_values(
    include: &Path,
    headers: &[&str],
    typedef_name: &str,
    work: &Path,
    tag: &str,
) -> Result<Vec<(String, i64)>, String> {
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let source = work.join(format!("{tag}.i.c"));
    let text: String = headers.iter().map(|h| format!("#include <{h}>\n")).collect();
    std::fs::write(&source, text).map_err(|e| e.to_string())?;
    let out = gcc(include)
        .args(["-E", "-P"])
        .arg(&source)
        .output()
        .map_err(|e| format!("gcc: {e}"))?;
    if !out.status.success() {
        return Err(format!("gcc -E: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let names = enumerator_names(&String::from_utf8_lossy(&out.stdout), typedef_name)?;
    let probe = work.join(format!("{tag}.probe.c"));
    let mut code: String = headers.iter().map(|h| format!("#include <{h}>\n")).collect();
    code.push_str("#include <stdio.h>\nint main(void) {\n");
    for n in &names {
        code.push_str(&format!("    printf(\"{n} %lld\\n\", (long long)({n}));\n"));
    }
    code.push_str("    return 0;\n}\n");
    std::fs::write(&probe, code).map_err(|e| e.to_string())?;
    let exe = work.join(format!("{tag}.probe"));
    let built = gcc(include)
        .arg("-o")
        .arg(&exe)
        .arg(&probe)
        .output()
        .map_err(|e| format!("gcc: {e}"))?;
    if !built.status.success() {
        return Err(format!(
            "probe for {typedef_name}: {}",
            String::from_utf8_lossy(&built.stderr)
        ));
    }
    let run = Command::new(&exe)
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    let mut values = Vec::new();
    for line in String::from_utf8_lossy(&run.stdout).lines() {
        let (n, v) = line.split_once(' ').ok_or("bad probe line")?;
        values.push((n.to_string(), v.parse::<i64>().map_err(|e| e.to_string())?));
    }
    Ok(values)
}

/// Names of the enumerators of `typedef enum { ... } typedef_name;` in preprocessed text.
pub fn enumerator_names(text: &str, typedef_name: &str) -> Result<Vec<String>, String> {
    let needle = format!("}} {typedef_name};");
    let end = match text.find(&needle) {
        Some(end) => end,
        None => {
            // `typedef enum tag name;` with the body written elsewhere as `enum tag { ... };`.
            let tag = text
                .split("typedef enum ")
                .skip(1)
                .find_map(|rest| {
                    let (tag, tail) = rest.split_once(' ')?;
                    tail.trim_start()
                        .strip_prefix(typedef_name)?
                        .starts_with(';')
                        .then_some(tag)
                })
                .ok_or_else(|| format!("no `typedef enum ... {typedef_name}`"))?;
            let open = text
                .find(&format!("enum {tag} {{"))
                .ok_or_else(|| format!("no body for enum {tag}"))?;
            open + text[open..].find('}').ok_or("no closing brace")?
        }
    };
    let start = text[..end].rfind('{').ok_or("no opening brace")?;
    let mut names = Vec::new();
    for item in text[start + 1..end].split(',') {
        let name = item.split('=').next().unwrap_or("").trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aux_info_lines() {
        let text = "/* a.h:1:NC */ extern cublasStatus_t cublasCreate_v2 (cublasHandle_t *);\n\
                    /* a.h:2:NC */ extern const char * nppGetGpuName (void);\n\
                    /* a.h:3:NF */ static int helper (int a, void (*f)(int, int));\n";
        let map = parse_aux_info(text);
        assert_eq!(map["cublasCreate_v2"].ret, "cublasStatus_t");
        assert_eq!(map["cublasCreate_v2"].params, "cublasHandle_t *");
        assert_eq!(map["nppGetGpuName"].ret, "const char *");
        assert!(split_params(&map["nppGetGpuName"].params).is_empty());
        assert!(map["helper"].inline);
        assert_eq!(split_params(&map["helper"].params).len(), 2);
    }

    #[test]
    fn sizes_from_the_spelling() {
        assert_eq!(builtin_size("const char *"), Some(8));
        assert_eq!(builtin_size("unsigned int"), Some(4));
        assert_eq!(builtin_size("long long unsigned int"), Some(8));
        assert_eq!(builtin_size("double"), Some(8));
        assert_eq!(builtin_size("NppiSize"), None);
    }

    #[test]
    fn splits_params_outside_parentheses() {
        assert_eq!(
            split_params("int, void (*)(int, int), const char *"),
            ["int", "void (*)(int, int)", "const char *"]
        );
        assert!(split_params("void").is_empty());
    }

    #[test]
    fn enumerators_of_a_typedef() {
        let text = "typedef enum { A = -3, B, C = 7 } Status_t; typedef enum { X } Other;";
        assert_eq!(enumerator_names(text, "Status_t").unwrap(), ["A", "B", "C"]);
        let tagged = "enum st { P = 1, Q }; typedef enum st st_t;";
        assert_eq!(enumerator_names(tagged, "st_t").unwrap(), ["P", "Q"]);
    }
}
