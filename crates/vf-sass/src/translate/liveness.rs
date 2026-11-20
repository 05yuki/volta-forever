//! Predicate liveness: which predicates (ordinary and uniform) hold a value that may
//! still be read, before and after each instruction. Used by `stand_ins` to assign
//! ordinary predicates to uniform ones.

use crate::decode::{LETTER_P, LETTER_Q};
use crate::field;
use crate::generated::fields::*;

use super::parse::FuncData;
use super::util::find_relative;

// ---- Liveness ----

/// Which predicates hold a value that may still be read, before and after each instruction: bits 0
/// to 6 the ordinary ones, bits 7 to 13 the uniform ones; and where each instruction leads.
pub(super) struct Liveness {
    pub(super) live_in: Vec<u64>,
    pub(super) live_out: Vec<u64>,
    pub(super) defs: Vec<u64>,
    pub(super) uses: Vec<u64>,
    pub(super) kills: Vec<u64>,
    pub(super) external: Vec<bool>,
    pub(super) successors: Vec<Vec<usize>>,
}

pub(super) fn compute_liveness(fd: &FuncData) -> Liveness {
    let n = fd.src.len();
    let returns_to: Vec<usize> = fd
        .src
        .iter()
        .enumerate()
        .filter(|(i, d)| d.name.starts_with("call_rel") && *i + 1 < n)
        .map(|(i, _)| i + 1)
        .collect();

    let mut defs = vec![0u64; n];
    let mut uses = vec![0u64; n];
    let mut external = vec![false; n];
    let mut kills = vec![0u64; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut ends = vec![0u64; n];
    let everything: u64 = (1 << 14) - 1;

    for (i, d) in fd.src.iter().enumerate() {
        let pg_true = d.op_or(OP_PG, 7) == 7 && d.op_or(OP_PG_AT_NOT, 0) == 0;
        let upg_true = d.op_or(OP_UPG, 7) == 7 && d.op_or(OP_UPG_AT_NOT, 0) == 0;
        let unguarded = pg_true && upg_true;

        for &(letter, reg, _) in &d.reads {
            match letter {
                LETTER_P => {
                    uses[i] |= 1 << reg;
                }
                LETTER_Q => {
                    uses[i] |= 1 << (reg + 7);
                }
                _ => {}
            }
        }
        for &(letter, reg) in &d.writes {
            match letter {
                LETTER_P => {
                    defs[i] |= 1 << reg;
                }
                LETTER_Q => {
                    defs[i] |= 1 << (reg + 7);
                }
                _ => {}
            }
        }
        // The predicates that P2R reads and R2P writes are in a mask: a number, or else any.
        // (What R2P writes is not taken to end the life of what the predicate held: nvdisasm,
        // which validate.py checks the predicates against, does not take it so either.)
        let mask = if matches!(d.name, "p2r__RsIR_RIR" | "r2p__RIR") {
            d.op_or(OP_SB, 0x7f) as u64 & 0x7f
        } else {
            0x7f
        };
        if d.mnemonic == "P2R" {
            uses[i] |= mask;
        }
        if d.mnemonic == "R2P" {
            defs[i] |= mask;
        } else if unguarded {
            kills[i] = defs[i];
        }

        let rel = find_relative(d, 75);
        let target = rel.map(|(fid, scale)| (16 * i as i64 + 16 + field::signed(fid, d.word) * scale as i64) as u64);
        let mut falls = true;
        if d.mnemonic == "EXIT" && unguarded {
            falls = false;
        } else if d.name == "ret__REL" {
            falls = !unguarded;
            successors[i].extend_from_slice(&returns_to);
        } else if d.name == "brx_"
            && let Some(places) = fd.table_at(16 * i as u64)
        {
            // To one of the places its table lists.
            falls = !unguarded;
            successors[i].extend(
                places
                    .iter()
                    .filter(|&&t| t % 16 == 0 && t < (16 * n) as u64)
                    .map(|&t| (t / 16) as usize),
            );
        } else if d.mnemonic == "RET" || matches!(d.mnemonic, "BRX" | "JMX" | "JMP") {
            falls = !unguarded;
            ends[i] = everything;
        } else if d.mnemonic == "BRA" && unguarded {
            // (BRA.DIV and BRA.CONV go by whether the threads of the warp are together: the code
            // after them is where the others go. Taken as always taken until 2026-11-17: the
            // helper that a warp calls when it is not together was left out of the lives.)
            let pp_true = d.op_or(OP_PP, 7) == 7 && d.op_or(OP_PP_AT_NOT, 0) == 0;
            if pp_true && d.op_or(OP_COND, 0) == 0 {
                falls = false;
            }
        } else if d.name.starts_with("call_abs") {
            external[i] = true;
        }

        if falls && i + 1 < n {
            successors[i].push(i + 1);
        }
        if let Some(t) = target
            && d.name != "ret__REL"
            && d.mnemonic != "RET"
            && !matches!(d.mnemonic, "BRX" | "JMX" | "JMP")
            && t % 16 == 0
            && t < (16 * n) as u64
        {
            successors[i].push((t / 16) as usize);
        }
    }

    let mut live_in = vec![0u64; n];
    let mut live_out: Vec<u64> = ends.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..n).rev() {
            let mut out = ends[i];
            for &s in &successors[i] {
                out |= live_in[s];
            }
            let into = uses[i] | (out & !kills[i]);
            if out != live_out[i] || into != live_in[i] {
                live_out[i] = out;
                live_in[i] = into;
                changed = true;
            }
        }
    }
    Liveness {
        live_in,
        live_out,
        defs,
        uses,
        kills,
        external,
        successors,
    }
}
