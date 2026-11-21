//! Shared CTA barriers with a fixed number of arrivals per phase.

use std::fmt::Write as _;

use super::Unsupported;
use super::body::Body;
use super::types::{integer, place_of};
use crate::scan;

impl Body<'_> {
    pub(super) fn cta_barrier(&mut self, guard: &str, words: &[&str], operands: &[&str]) -> Result<(), Unsupported> {
        let error = |why: &str| Unsupported(format!("{}: {why}", words.join(".")));
        // For an initialized barrier with count N, the low word counts arrivals modulo 2N.
        // Progress in [0,N) is phase 0, progress in [N,2N) phase 1. The high word (N) is
        // immutable until reinitialization. Only the low word is ever accessed concurrently.
        let high_word = |address: &str| {
            let (base, offset) = place_of(address)?;
            Some(format!("[{base}+{}]", offset.checked_add(4)?))
        };
        let bad = || error("operands not recognised");
        let source = |s: &str| scan::register(s).is_some() || integer(s).is_some();
        match words {
            ["mbarrier", "init", "shared" | "shared::cta", "b64"] => {
                let &[address, count] = operands else { return Err(bad()) };
                let high = high_word(address).filter(|_| source(count)).ok_or_else(bad)?;
                emit!(self, "{guard}st.shared.u32 {high}, {count};");
                emit!(self, "{guard}st.shared.u32 {address}, 0;");
            }
            ["mbarrier", "arrive", "shared" | "shared::cta", "b64"]
            | [
                "cp",
                "async",
                "mbarrier",
                "arrive",
                "noinc",
                "shared" | "shared::cta",
                "b64",
            ] => {
                let (state, address) = match (words[0], operands) {
                    ("mbarrier", &[state, address]) if state == "_" || scan::register(state).is_some() => {
                        (Some(state), address)
                    }
                    ("cp", &[address]) => (None, address),
                    _ => return Err(bad()),
                };
                let high = high_word(address).ok_or_else(bad)?;
                let [count, limit, old] = self.fresh_n();
                emit!(self, "{guard}ld.shared.u32 {count}, {high};");
                emit!(self, "{guard}mad.lo.u32 {limit}, {count}, 2, -1;");
                // Each arrival takes/relinquishes the same CTA atomic, carrying the earlier
                // writers' releases through the last arrival. A noinc arrive accounts for one
                // arrival after this thread's copies, which Volta lowering already made synchronous.
                emit!(
                    self,
                    "{guard}atom.acq_rel.cta.shared.inc.u32 {old}, {address}, {limit};"
                );
                if let Some(state) = state.filter(|s| *s != "_") {
                    self.written(state);
                    emit!(self, "{guard}mov.b64 {state}, {{{old}, {count}}};");
                }
            }
            ["mbarrier", "test_wait", "parity", "shared" | "shared::cta", "b64"] => {
                let &[ready, address, parity] = operands else {
                    return Err(bad());
                };
                let high = high_word(address)
                    .filter(|_| scan::register(ready).is_some() && source(parity))
                    .ok_or_else(bad)?;
                let [count, progress, phase] = self.fresh_n();
                self.written(ready);
                emit!(self, "{guard}ld.shared.u32 {count}, {high};");
                emit!(self, "{guard}ld.acquire.cta.shared.u32 {progress}, {address};");
                emit!(self, "{guard}set.ge.u32.u32 {phase}, {progress}, {count};");
                emit!(self, "{guard}neg.s32 {phase}, {phase};");
                emit!(self, "{guard}setp.ne.u32 {ready}, {phase}, {parity};");
            }
            _ => return Err(error("no Volta sequence for this form")),
        }
        Ok(())
    }
}
