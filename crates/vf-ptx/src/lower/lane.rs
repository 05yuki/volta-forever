//! Values derived from the warp lane: computed once at the top of the function and used
//! throughout the lowered body for shuffle indices and address computations.

/// Values derived from the lane, computed once at the top of the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Lane {
    /// The lane (0..31).
    Id,
    /// Thread in group: `lane & 3`.
    Thread,
    /// Group: `lane >> 2`.
    Group,
    /// `lane & mask`, for a single-bit mask.
    Bit(u32),
    /// Predicate: the lane has bit `mask` set.
    Has(u32),
    /// Lane that holds, for the FP16 mma, the B elements this thread provides to its octet, for the
    /// columns 0..3 (`false`) or 4..7 (`true`) of B: `(lane >> 4) * 8 + thread`, plus 16.
    Columns(bool),
    /// First lane of the group: `lane & 28`.
    GroupStart,
    /// Lane `n` (0..3) of the group.
    InGroup(u32),
    /// `thread * 8`: first lane of the group whose number is twice the thread's.
    Thread8,
    /// Lane `n` (0..7) counted from `Thread8`: the groups `2 * thread` and `2 * thread + 1`.
    InPair(u32),
    /// Lane that provides the address of this thread's row of matrix `n` to `ldmatrix`: `8 * n + group`.
    RowOf(u32),
    /// Byte offset of this thread's pair of elements in a row: `4 * thread`.
    PairOffset,
    /// `thread * 2`.
    Thread2,
    /// For `ldmatrix.trans`, lane that provides the address of row `2 * thread + n % 2` of matrix
    /// `n / 2`.
    ColumnOf(u32),
    /// Byte offset of column `group` in a row: `2 * group`.
    ColumnOffset,
    /// For `movmatrix`, the lanes that hold the elements `(2 * thread, group)` and
    /// `(2 * thread + 1, group)`: `8 * thread + group / 2`, plus 4.
    Transposed(bool),
    /// Shift that brings the element of column `group` to the low half: `16 * (group & 1)`.
    TransposedShift,
    /// Half of the warp: `lane >> 4`.
    Half,
    /// The other half: `(lane >> 4) ^ 1`.
    OtherHalf,
    /// `2 * half`.
    Half2,
    /// Lane that gives `ldmatrix` the address of the row of matrix `n` that holds the column of B
    /// this thread provides to its octet: `8 * n + 4 * (thread >> 1) + 2 * half + (thread & 1)`.
    ColumnRow(u32),
    /// Lane that gives `ldmatrix` the address of the row of the thread's group in matrix `.0`
    /// (threads 0 and 1) or `.1` (threads 2 and 3): the row of A this thread provides to its octet.
    PairRow(u32, u32),
    /// For `ldmatrix.trans`, lane that gives the address of row `2 * thread + half` of matrix `.0`, or
    /// `2 * thread + (half ^ 1)` when `.1` is set: the k this thread provides in each Volta instruction.
    PassRow(u32, bool),
    /// Byte offset of the four columns the half provides, in such a row: `4 * half`.
    HalfOffset,
    /// `2 * ((lane >> 3) & 1)`: twice the k-half index for the chain form.
    KHalf2,
    /// For the chain form: the lane whose Turing register holds the first word of the row or column
    /// this thread needs. `4 * thread + (lane & 16) + KHalf2`.
    ChainW0,
    /// ChainW0 + 1.
    ChainW1,
    /// For the chain form with direct loading: the lane whose address register holds the shared-memory
    /// address of the row this thread's octet needs. `(lane & 3) | ((lane & 4) << 1) | ((lane >> 2) & 4)`.
    /// Added to `8 * first_matrix_index` at the `ldmatrix` site for the right fragment.
    ChainARow0,
    /// The column this thread's octet reads in B: `(lane & 3) | ((lane >> 2) & 4)`.
    ChainBCol,
    /// For the end of a chain: the lane whose partial sums make registers 0 and 1 of this
    /// thread's accumulator, `(lane & 24) | ((lane >> 2) & 1) | ((lane & 1) << 1)`; for registers
    /// 2 and 3 (`true`), the same plus 4.
    ChainOut(bool),
}

impl Lane {
    pub(super) fn name(self) -> String {
        match self {
            Lane::Id => "%vf_l".to_string(),
            Lane::Thread => "%vf_k".to_string(),
            Lane::Group => "%vf_g".to_string(),
            Lane::Bit(mask) => format!("%vf_b{mask}"),
            Lane::Has(mask) => format!("%vf_p{mask}"),
            Lane::Columns(upper) => format!("%vf_s{}", upper as u32),
            Lane::GroupStart => "%vf_ia0".to_string(),
            Lane::InGroup(n) => format!("%vf_ia{n}"),
            Lane::Thread8 => "%vf_ib0".to_string(),
            Lane::InPair(n) => format!("%vf_ib{n}"),
            Lane::RowOf(0) => Lane::Group.name(),
            Lane::RowOf(n) => format!("%vf_lm{n}"),
            Lane::PairOffset => "%vf_o4".to_string(),
            Lane::Thread2 => "%vf_lt0".to_string(),
            Lane::ColumnOf(n) => format!("%vf_lt{n}"),
            Lane::ColumnOffset => "%vf_o2".to_string(),
            Lane::Transposed(odd) => format!("%vf_v{}", odd as u32),
            Lane::TransposedShift => "%vf_vs".to_string(),
            Lane::Half => "%vf_h".to_string(),
            Lane::OtherHalf => "%vf_hn".to_string(),
            Lane::Half2 => "%vf_h2".to_string(),
            Lane::ColumnRow(n) => format!("%vf_cr{n}"),
            Lane::PairRow(first, second) => format!("%vf_pr{first}{second}"),
            Lane::PassRow(n, other) => format!("%vf_tr{n}{}", other as u32),
            Lane::HalfOffset => "%vf_oh".to_string(),
            Lane::KHalf2 => "%vf_kh2".to_string(),
            Lane::ChainW0 => "%vf_cw0".to_string(),
            Lane::ChainW1 => "%vf_cw1".to_string(),
            Lane::ChainARow0 => "%vf_car0".to_string(),
            Lane::ChainBCol => "%vf_cbc".to_string(),
            Lane::ChainOut(upper) => format!("%vf_co{}", upper as u32),
        }
    }

    /// The same value under another variant, if there is one.
    pub(super) fn canonical(self) -> Lane {
        match self {
            Lane::InGroup(0) => Lane::GroupStart,
            Lane::InPair(0) => Lane::Thread8,
            Lane::RowOf(0) => Lane::Group,
            Lane::ColumnOf(0) => Lane::Thread2,
            Lane::PairRow(first, second) if first == second => Lane::RowOf(first).canonical(),
            other => other,
        }
    }

    pub(super) fn needs(self) -> Vec<Lane> {
        match self {
            Lane::Id => vec![],
            Lane::Thread | Lane::Group | Lane::Bit(_) | Lane::GroupStart | Lane::TransposedShift => vec![Lane::Id],
            Lane::Has(mask) => vec![Lane::Bit(mask)],
            Lane::Columns(false) => vec![Lane::Id, Lane::Thread],
            Lane::Columns(true) => vec![Lane::Columns(false)],
            Lane::InGroup(_) => vec![Lane::GroupStart],
            Lane::Thread8 | Lane::PairOffset | Lane::Thread2 => vec![Lane::Thread],
            Lane::InPair(_) => vec![Lane::Thread8],
            Lane::RowOf(_) | Lane::ColumnOffset => vec![Lane::Group],
            Lane::ColumnOf(_) => vec![Lane::Thread2],
            Lane::Transposed(false) => vec![Lane::Id, Lane::Thread8],
            Lane::Transposed(true) => vec![Lane::Transposed(false)],
            Lane::Half => vec![Lane::Id],
            Lane::OtherHalf | Lane::Half2 | Lane::HalfOffset => vec![Lane::Half],
            Lane::ColumnRow(0) => vec![Lane::Id, Lane::Half2, Lane::Bit(1)],
            Lane::ColumnRow(_) => vec![Lane::ColumnRow(0)],
            Lane::PairRow(first, second) => vec![Lane::RowOf(first), Lane::RowOf(second), Lane::Has(2)],
            Lane::PassRow(_, other) => vec![Lane::Thread2, if other { Lane::OtherHalf } else { Lane::Half }],
            Lane::KHalf2 => vec![Lane::Bit(8)],
            Lane::ChainW0 => vec![Lane::Thread, Lane::Bit(16), Lane::KHalf2],
            Lane::ChainW1 => vec![Lane::ChainW0],
            Lane::ChainARow0 => vec![Lane::ChainBCol, Lane::Bit(4)],
            Lane::ChainBCol => vec![Lane::Half, Lane::Thread],
            Lane::ChainOut(false) => vec![Lane::Bit(4), Lane::Bit(16), Lane::Bit(8), Lane::Bit(1)],
            Lane::ChainOut(true) => vec![Lane::ChainOut(false)],
        }
    }

    /// The instructions that compute the value into its register.
    pub(super) fn code(self) -> String {
        let d = self.name();
        let (l, k, g) = (Lane::Id.name(), Lane::Thread.name(), Lane::Group.name());
        match self {
            Lane::Id => format!("mov.u32 {d}, %laneid;"),
            Lane::Thread => format!("and.b32 {d}, {l}, 3;"),
            Lane::Group => format!("shr.u32 {d}, {l}, 2;"),
            Lane::Bit(mask) => format!("and.b32 {d}, {l}, {mask};"),
            Lane::Has(mask) => format!("setp.ne.b32 {d}, {}, 0;", Lane::Bit(mask).name()),
            Lane::Columns(false) => format!("shr.u32 {d}, {l}, 1;\n\tand.b32 {d}, {d}, 8;\n\tor.b32 {d}, {d}, {k};"),
            Lane::Columns(true) => format!("or.b32 {d}, {}, 16;", Lane::Columns(false).name()),
            Lane::GroupStart => format!("and.b32 {d}, {l}, 28;"),
            Lane::InGroup(n) => format!("or.b32 {d}, {}, {n};", Lane::GroupStart.name()),
            Lane::Thread8 => format!("shl.b32 {d}, {k}, 3;"),
            Lane::InPair(n) => format!("or.b32 {d}, {}, {n};", Lane::Thread8.name()),
            Lane::RowOf(n) => format!("or.b32 {d}, {g}, {};", 8 * n),
            Lane::PairOffset => format!("shl.b32 {d}, {k}, 2;"),
            Lane::Thread2 => format!("shl.b32 {d}, {k}, 1;"),
            Lane::ColumnOf(n) => format!("or.b32 {d}, {}, {};", Lane::Thread2.name(), 8 * (n / 2) + n % 2),
            Lane::ColumnOffset => format!("shl.b32 {d}, {g}, 1;"),
            Lane::Transposed(false) => format!("shr.u32 {d}, {l}, 3;\n\tor.b32 {d}, {d}, {};", Lane::Thread8.name()),
            Lane::Transposed(true) => format!("or.b32 {d}, {}, 4;", Lane::Transposed(false).name()),
            Lane::TransposedShift => format!("and.b32 {d}, {l}, 4;\n\tshl.b32 {d}, {d}, 2;"),
            Lane::Half => format!("shr.u32 {d}, {l}, 4;"),
            Lane::OtherHalf => format!("xor.b32 {d}, {}, 1;", Lane::Half.name()),
            Lane::Half2 => format!("shl.b32 {d}, {}, 1;", Lane::Half.name()),
            Lane::ColumnRow(0) => format!(
                "and.b32 {d}, {l}, 2;\n\tshl.b32 {d}, {d}, 1;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};",
                Lane::Half2.name(),
                Lane::Bit(1).name()
            ),
            Lane::ColumnRow(n) => format!("or.b32 {d}, {}, {};", Lane::ColumnRow(0).name(), 8 * n),
            Lane::PairRow(first, second) => format!(
                "selp.b32 {d}, {}, {}, {};",
                Lane::RowOf(second).canonical().name(),
                Lane::RowOf(first).canonical().name(),
                Lane::Has(2).name()
            ),
            Lane::PassRow(n, other) => {
                let half = if other { Lane::OtherHalf } else { Lane::Half };
                let row = format!("or.b32 {d}, {}, {};", Lane::Thread2.name(), half.name());
                if n == 0 {
                    row
                } else {
                    format!("{row}\n\tor.b32 {d}, {d}, {};", 8 * n)
                }
            }
            Lane::HalfOffset => format!("shl.b32 {d}, {}, 2;", Lane::Half.name()),
            Lane::KHalf2 => format!("shr.u32 {d}, {}, 2;", Lane::Bit(8).name()),
            Lane::ChainW0 => {
                // 4*thread + (lane & 16) + KHalf2 = ((lane & 3) << 2) | (lane & 16) | KHalf2
                format!(
                    "shl.b32 {d}, {}, 2;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};",
                    Lane::Thread.name(),
                    Lane::Bit(16).name(),
                    Lane::KHalf2.name(),
                )
            }
            Lane::ChainW1 => format!("or.b32 {d}, {}, 1;", Lane::ChainW0.name()),
            Lane::ChainARow0 => {
                // (lane & 3) | ((lane & 4) << 1) | ((lane >> 2) & 4)
                // = ChainBCol | (Bit(4) << 1), where ChainBCol = thread | (half << 2).
                format!(
                    "shl.b32 {d}, {}, 1;\n\tor.b32 {d}, {d}, {};",
                    Lane::Bit(4).name(),
                    Lane::ChainBCol.name(),
                )
            }
            Lane::ChainBCol => {
                // (lane & 3) | ((lane >> 2) & 4) = thread | (half << 2)
                format!(
                    "shl.b32 {d}, {}, 2;\n\tor.b32 {d}, {d}, {};",
                    Lane::Half.name(),
                    Lane::Thread.name(),
                )
            }
            Lane::ChainOut(false) => {
                // ((lane & 4) >> 2) | (lane & 16) | (lane & 8) | ((lane & 1) << 1): the last one
                // added twice, into a bit that is clear.
                let one = Lane::Bit(1).name();
                format!(
                    "shr.u32 {d}, {}, 2;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};\n\tadd.u32 {d}, {d}, {one};\n\tadd.u32 {d}, {d}, {one};",
                    Lane::Bit(4).name(),
                    Lane::Bit(16).name(),
                    Lane::Bit(8).name(),
                )
            }
            Lane::ChainOut(true) => format!("or.b32 {d}, {}, 4;", Lane::ChainOut(false).name()),
        }
    }
}
