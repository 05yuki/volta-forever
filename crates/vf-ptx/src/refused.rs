//! What the driver's compiler refuses in a module after its adaptation.
//!
//! `lower_to_volta` rewrites or traps the instructions it knows. A module can still hold something the
//! compiler for Volta refuses, and the driver then refuses the module as a whole. Its messages name the
//! lines: the functions that hold them are replaced by traps (and the function directives it names
//! removed), after which the rest of the module compiles. The kernels that are, or call, such a
//! function must not be given to the program: launched, they would end the CUDA context.

use std::collections::HashMap;

use crate::deps::ModuleInfo;
use crate::{TRAP_BODY, functions};

/// A message of the compiler about one line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerError {
    /// The line in the text, counted from 1.
    pub line: usize,
    /// The message, e.g. "Feature 'discard' requires .target sm_80 or higher".
    pub message: String,
}

/// The lines a compiler log names as errors. The driver writes
/// `ptxas application ptx input, line 287; error   : MESSAGE`, `ptxas` the same with a file name; what
/// stops the parser is `fatal` instead of `error`.
pub fn parse_error_lines(log: &str) -> Vec<CompilerError> {
    log.lines()
        .filter_map(|line| {
            let (_, rest) = line.split_once(", line ")?;
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            let number = rest[..digits].parse().ok()?;
            let rest = rest[digits..].strip_prefix(';')?.trim_start();
            let rest = rest.strip_prefix("error").or_else(|| rest.strip_prefix("fatal"))?;
            let message = rest.trim_start().strip_prefix(':')?.trim();
            Some(CompilerError {
                line: number,
                message: message.to_string(),
            })
        })
        .collect()
}

/// A function made to trap because the compiler refused lines of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusedFunction {
    pub name: String,
    /// The compiler's messages, without repetition.
    pub reasons: Vec<String>,
}

/// A kernel that must not be given to the program: `function`, which it is or calls, was made to trap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    pub kernel: String,
    pub function: String,
    /// Why the function traps: the instruction without a sequence, or the compiler's message.
    pub reason: String,
}

/// Where each line of `text` starts.
fn line_starts(text: &[u8]) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(
        text.iter()
            .enumerate()
            .filter(|&(at, &byte)| byte == b'\n' && at + 1 < text.len())
            .map(|(at, _)| at + 1),
    );
    starts
}

/// `text` with every function that holds one of the lines of `errors` made to trap. A line among the
/// directives of a function (between its parameters and its body) is removed and the function trapped:
/// a kernel cannot do without what it asked for there.
///
/// `None` when a line belongs to no function, or is one of its name and parameters: nothing here
/// makes such a module compile.
pub fn trap_functions_at_lines(text: &[u8], errors: &[CompilerError]) -> Option<(Vec<u8>, Vec<RefusedFunction>)> {
    let found = functions(text);
    let starts = line_starts(text);
    // By function: the directive lines to remove and the messages.
    let mut refused: HashMap<usize, (Vec<std::ops::Range<usize>>, Vec<String>)> = HashMap::new();
    for error in errors {
        let start = *starts.get(error.line.checked_sub(1)?)?;
        let end = starts.get(error.line).copied().unwrap_or(text.len());
        let index = found
            .iter()
            .position(|function| function.name.end <= start && start < function.body.end)?;
        let function = &found[index];
        let (lines, reasons) = refused.entry(index).or_default();
        if start < function.body.start {
            let line = &text[start..end];
            let directive = line.iter().find(|byte| !byte.is_ascii_whitespace()) == Some(&b'.')
                && !line.trim_ascii_start().starts_with(b".param")
                && end <= function.body.start;
            if !directive {
                return None;
            }
            if !lines.contains(&(start..end)) {
                lines.push(start..end);
            }
        }
        if !reasons.contains(&error.message) {
            reasons.push(error.message.clone());
        }
    }
    let mut out = Vec::with_capacity(text.len());
    let mut functions_refused = Vec::new();
    let mut copied = 0;
    for (index, function) in found.iter().enumerate() {
        let Some((mut lines, reasons)) = refused.remove(&index) else {
            continue;
        };
        lines.sort_by_key(|line| line.start);
        for line in lines {
            out.extend_from_slice(&text[copied..line.start]);
            copied = line.end;
        }
        out.extend_from_slice(&text[copied..function.body.start]);
        out.extend_from_slice(TRAP_BODY);
        copied = function.body.end;
        functions_refused.push(RefusedFunction {
            name: String::from_utf8_lossy(&text[function.name.clone()]).into_owned(),
            reasons,
        });
    }
    out.extend_from_slice(&text[copied..]);
    Some((out, functions_refused))
}

/// The kernels of `info` that are, or call, one of the functions of `trapped` (name and reason), with
/// the first such function each reaches.
pub fn kernels_reaching(info: &ModuleInfo, trapped: &HashMap<&str, &str>) -> Vec<Refused> {
    info.reach()
        .filter_map(|(kernel, reached)| {
            let function = reached.iter().find(|name| trapped.contains_key(name.as_str()))?;
            Some(Refused {
                kernel: kernel.to_string(),
                function: function.clone(),
                reason: trapped[function.as_str()].to_string(),
            })
        })
        .collect()
}

/// A module without what the compiler refused in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Removal {
    /// The text to compile instead.
    pub text: Vec<u8>,
    /// The functions made to trap.
    pub functions: Vec<RefusedFunction>,
    /// The kernels that are or call one of them.
    pub kernels: Vec<Refused>,
}

/// `text` without the functions that the compiler's `log` refuses. `None` when the log names no line
/// of a function: the module cannot be made to compile this way.
pub fn without_refused(text: &[u8], log: &str) -> Option<Removal> {
    let errors = parse_error_lines(log);
    let (new_text, functions) = trap_functions_at_lines(text, &errors)?;
    if functions.is_empty() {
        return None;
    }
    // A few messages tell why; a function can have one per line.
    let reasons: Vec<String> = functions
        .iter()
        .map(|function| function.reasons[..function.reasons.len().min(3)].join("; "))
        .collect();
    let trapped: HashMap<&str, &str> = functions
        .iter()
        .zip(&reasons)
        .map(|(function, reason)| (function.name.as_str(), reason.as_str()))
        .collect();
    let kernels = kernels_reaching(&ModuleInfo::analyse(text), &trapped);
    Some(Removal {
        text: new_text,
        functions,
        kernels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_messages_of_the_driver_and_of_ptxas() {
        let log = "ptxas application ptx input, line 287; error   : Modifier '.NaN' requires .target sm_80 or higher\n\
                   ptxas application ptx input, line 300; error   : Feature 'discard' requires .target sm_80 or higher\n\
                   ptxas fatal   : Ptx assembly aborted due to errors\n";
        assert_eq!(
            parse_error_lines(log),
            [
                CompilerError {
                    line: 287,
                    message: "Modifier '.NaN' requires .target sm_80 or higher".to_string()
                },
                CompilerError {
                    line: 300,
                    message: "Feature 'discard' requires .target sm_80 or higher".to_string()
                },
            ]
        );
        let log = "ptxas /dev/stdin, line 7; error   : Unknown modifier '.new'\n\
                   ptxas /dev/stdin, line 7; fatal   : Parsing error near '<': syntax error\n\
                   ptxas fatal   : Ptx assembly aborted due to errors\n";
        let errors = parse_error_lines(log);
        assert_eq!(errors.len(), 2);
        assert_eq!(
            (errors[1].line, errors[1].message.as_str()),
            (7, "Parsing error near '<': syntax error")
        );
        assert!(parse_error_lines("").is_empty());
        assert!(parse_error_lines("ptxas fatal   : Ptx assembly aborted\n").is_empty());
        // A warning names a line too and is not an error.
        assert!(parse_error_lines("ptxas a.ptx, line 3; warning : Unused register\n").is_empty());
    }

    const MODULE: &str = "\
.version 8.8\n\
.target sm_70\n\
.address_size 64\n\
\n\
.entry good_kernel()\n\
{\n\
\tret;\n\
}\n\
\n\
.entry bad_kernel()\n\
.maxntid 256, 1, 1\n\
.explicitcluster\n\
.reqnctapercluster 2, 1, 1\n\
{\n\
\t.reg .f32 %f<4>;\n\
\tmin.NaN.f32 %f1, %f2, %f3;\n\
\tret;\n\
}\n\
\n\
.func helper()\n\
{\n\
\tdiscard.global.L2 [0], 128;\n\
\tret;\n\
}\n\
\n\
.entry caller()\n\
{\n\
\tcall helper, ();\n\
\tret;\n\
}\n";

    fn error(line: usize, message: &str) -> CompilerError {
        CompilerError {
            line,
            message: message.to_string(),
        }
    }

    #[test]
    fn traps_the_function_that_holds_a_refused_line() {
        let (text, refused) =
            trap_functions_at_lines(MODULE.as_bytes(), &[error(16, "A"), error(17, "B"), error(16, "A")]).unwrap();
        let text = String::from_utf8(text).unwrap();
        assert_eq!(
            refused,
            [RefusedFunction {
                name: "bad_kernel".to_string(),
                reasons: vec!["A".to_string(), "B".to_string()]
            }]
        );
        assert!(text.contains(".entry good_kernel()\n{\n\tret;\n}"));
        assert!(text.contains(".reqnctapercluster 2, 1, 1\n{\n\ttrap;\n\tret;\n}"));
        assert!(text.contains(".func helper()\n{\n\tdiscard.global.L2 [0], 128;\n\tret;\n}"));
    }

    #[test]
    fn removes_a_refused_directive_and_traps_its_function() {
        let (text, refused) = trap_functions_at_lines(MODULE.as_bytes(), &[error(12, "C"), error(13, "D")]).unwrap();
        let text = String::from_utf8(text).unwrap();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].name, "bad_kernel");
        assert!(text.contains(".entry bad_kernel()\n.maxntid 256, 1, 1\n{\n\ttrap;\n\tret;\n}"));
    }

    #[test]
    fn gives_up_on_a_line_outside_the_functions() {
        // The header, the line of a name, a line beyond the text.
        for line in [1, 10, 400, 0] {
            assert!(trap_functions_at_lines(MODULE.as_bytes(), &[error(line, "E")]).is_none());
        }
        let (text, refused) = trap_functions_at_lines(MODULE.as_bytes(), &[]).unwrap();
        assert_eq!(text, MODULE.as_bytes());
        assert!(refused.is_empty());
    }

    #[test]
    fn names_the_kernels_that_reach_a_refused_function() {
        let log = "ptxas application ptx input, line 22; error   : Feature 'discard' requires .target sm_80 or higher\n\
                   ptxas application ptx input, line 16; error   : Modifier '.NaN' requires .target sm_80 or higher\n";
        let removal = without_refused(MODULE.as_bytes(), log).unwrap();
        let names: Vec<&str> = removal
            .functions
            .iter()
            .map(|function| function.name.as_str())
            .collect();
        assert_eq!(names, ["bad_kernel", "helper"]);
        assert_eq!(
            removal.kernels,
            [
                Refused {
                    kernel: "bad_kernel".to_string(),
                    function: "bad_kernel".to_string(),
                    reason: "Modifier '.NaN' requires .target sm_80 or higher".to_string()
                },
                Refused {
                    kernel: "caller".to_string(),
                    function: "helper".to_string(),
                    reason: "Feature 'discard' requires .target sm_80 or higher".to_string()
                },
            ]
        );
        // The caller keeps its body: it is refused when the program asks for it.
        let text = String::from_utf8(removal.text).unwrap();
        assert!(text.contains(".entry caller()\n{\n\tcall helper, ();\n\tret;\n}"));
        assert!(text.contains(".func helper()\n{\n\ttrap;\n\tret;\n}"));
        // Nothing to remove: no next attempt.
        assert!(
            without_refused(
                MODULE.as_bytes(),
                "ptxas fatal   : Ptx assembly aborted due to errors\n"
            )
            .is_none()
        );
        assert!(without_refused(MODULE.as_bytes(), "ptxas application ptx input, line 2; error   : E\n").is_none());
    }
}
