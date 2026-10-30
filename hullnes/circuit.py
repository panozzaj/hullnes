"""The NES CPU as a circuit graph for a vanilla transformer with 2D hardmax attention heads.

Every position runs the same circuit and outputs ``next``, the id of the next trace token
(nesformer's format, see third_party/nesformer/spec.py). Building blocks (chipformer.dsl):

  lookup   a hardmax head with query (q, 1) and key (kx, ky): at position t it copies the values
           of the s <= t that maximizes q * kx_s + ky_s. Every head is a "highest line at q"
           query, so decoding keeps an exact O(log n) cache per head (a Li Chao tree), the
           convex-hull trick of Percepta's HullKVCache in its dual form.
  cumsum   a running total: a uniform-attention head times the position (Percepta's
           instruction-pointer counter)
  ffn      ReGLU units ``coef * a * relu(b)`` with a, b linear in the residual stream

Memory is the trace itself (see tokens.py for the prompt):

  ROM head   PRG-ROM in the prompt, keyed by position (score 2 q s - s^2 peaks at s = q).
             Position 0 holds a zero byte, which is what a read of a non-ROM address gets.
  RAM head   keyed cells: work RAM 0..2047, the I/O port 2048, a zero cell 2049, PRG-RAM
             2050... Entries are the prompt's RI tokens, ST / RM / PSH writes (their key is
             computed one position earlier, where the model decided to write, and carried by a
             prev head), and IN tokens (the host's answers, key 2048). Score T (2 k q - k^2) + s:
             an exact key match dominates and recency breaks ties, so a read returns the latest
             write. A read of a ROM address queries the zero cell.

So the operand of every read is ROM + RAM, with no select. Registers and flags are "latest token
of family F" heads; the PC is anchor + bytes committed since the anchor, as in nesformer. A
mapper board's state is a few more latest heads; PRG banking is folded into the queries.

Layer plan:

  L1  attn  registers, flags, latches, board state, record start, NMI, GO; cumsum of lengths
      ffn   record stage, in-NMI flag, SP +- 1, prompt keys, (zp),Y address, register inc/dec
  L2  attn  PC anchor and the cumsum at the anchor
      ffn   fetch queries (banked) for PC, PC + 1, PC + 2; refresh test
  L3  attn  fetch: opcode (decoded by its embedding), operand bytes
      ffn   address candidates per mode, branch test, return address bytes
  L4  ffn   read address qa, EA bits for payloads, pushed byte, page-cross
  L5  ffn   RAM and ROM queries (banked), regions, write key (high and low byte)
  L6  attn  prev: write key and pushed byte of this token
      ffn   key^2 of this token's write; I/O register index; MMC1 register select
  L7  attn  RAM and ROM reads
      ffn   operand bits, adder, compares, inc/dec, jump targets
  L8  ffn   bitwise ops, overflow, BIT
  L9  ffn   BIT's Z flag
  L10 ffn   the next token
"""

from dataclasses import dataclass

from chipformer.dsl.graph import Expr, as_expr
from chipformer.dsl.lib import Circuit
from hullnes import tokens as T
from hullnes.mapper import BANK
from nesformer import spec as S
from nesformer.cpu import BRANCH_FLAG, LENGTH, OPS, PAGE_PENALTY

MAX_POS = 1 << 20  # positions must stay below this (recency term of the RAM head)
BIG = 1 << 18  # gate offset for piecewise-linear units (> any gated argument range)

MNS = sorted({mn for mn, _, _ in OPS.values()})
MODES = ["imp", "acc", "imm", "zp", "zpx", "zpy", "abs", "abx", "aby", "inx", "iny", "rel", "ind"]
DIRECT = ["imm", "zp", "zpx", "zpy", "abs", "abx", "aby"]
READS = ["LDA", "LDX", "LDY", "ADC", "SBC", "AND", "ORA", "EOR", "CMP", "CPX", "CPY", "BIT"]
STORES = ["STA", "STX", "STY"]
RMWS = ["ASL", "LSR", "ROL", "ROR", "INC", "DEC"]
FLAG_OPS = {"CLC": ("C", 0), "SEC": ("C", 1), "CLI": ("I", 0), "SEI": ("I", 1),
            "CLV": ("V", 0), "CLD": ("D", 0), "SED": ("D", 1)}


@dataclass(frozen=True)
class Config:
    board: int = 0  # iNES mapper number
    n_prg: int = 0x8000  # bytes of PRG-ROM
    prgram: bool = False  # 8 KB RAM at $6000-$7FFF

    @property
    def n16(self) -> int:
        return self.n_prg // BANK

    @property
    def layout(self) -> T.Layout:
        return T.Layout(self.n_prg, self.prgram)

    @property
    def io_end(self) -> int:
        return 0x6000 if self.prgram else 0x8000

    @property
    def k_null(self) -> int:
        return self.layout.n_ram  # a key nothing reads

    @classmethod
    def for_board(cls, board) -> "Config":
        return cls(board.number, len(board.prg), board.prgram)


def vlen(fam: str, p: int) -> int:
    """The instruction length a final token commits (spec.length_of on family/payload)."""
    if fam in ("A", "X", "Y"):
        return p >> 11
    if fam in ("S", "PSH"):
        return p >> 8
    if fam == "ST":
        return p >> 9
    if fam == "RM":
        return p >> 10
    if fam == "P":
        return p >> 7
    if fam == "NT":
        return p
    if fam == "IOW":
        return p >> 14
    if fam == "MAP":
        return p >> 11
    return 0


def vfinal(fam: str, p: int) -> int:
    return int(fam in ("J", "JI", "R", "JL", "RL", "HALT", "GO") or vlen(fam, p) > 0)


def base(fam: str) -> int:
    return S.FAMILIES[fam][0]


class NesCircuit(Circuit):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ranges: list[tuple[Expr, int, int, str]] = []  # (expr, lo, hi): debug checks

    # --- building blocks -----------------------------------------------------------------------
    def tf(self, name: str, fn) -> Expr:
        """A token feature fn(family, payload) -> int (a column of the embedding)."""
        return self.token_feature(name, lambda t, fn=fn: int(fn(*T.family(t))))

    def latest(self, flag: Expr, values: list, name: str = "latest") -> list[Expr]:
        """Values at the latest position s <= t with flag(s) = 1 (flag in {0, 1})."""
        return self.lookup(0, 0, self.pos + 2 * MAX_POS * (flag - 1), values, (0, 0), name)

    def prev(self, values: list, name: str = "prev") -> list[Expr]:
        """Values at position t - 1: score 2 s (t-1) - s^2 peaks at s = t - 1."""
        return self.lookup(self.pos - 1, 2 * self.pos, -self.pos2, values, (-1, MAX_POS), name)

    @staticmethod
    def AND(lits) -> Expr:
        """relu(AND(lits)) is 1 when every literal (0/1) is 1, else 0."""
        lits = [as_expr(x) for x in lits]
        return sum(lits, Expr()) - (len(lits) - 1)

    def sel(self, cases, name: str) -> Expr:
        """sum(value * [all lits]) over mutually exclusive cases; one ReGLU unit per case."""
        return self.ffn([(1, as_expr(v), self.AND(lits)) for lits, v in cases], name)

    def pwl_terms(self, z: Expr, lo: int, hi: int, f, lits=None) -> tuple[list, int]:
        """Units of f(z) for an integer z in [lo, hi] (gated by lits): the piecewise linear
        interpolation of f, one ReGLU unit per slope change. Returns (terms, constant)."""
        gate = self.AND(lits) if lits else None
        off = BIG * (gate - 1) if gate is not None else Expr()
        terms = []
        if gate is not None and f(lo):
            terms.append((f(lo), self.one, gate))
        slope = 0
        for k in range(lo, hi):
            s = f(k + 1) - f(k)
            if s != slope:
                terms.append((s - slope, self.one, z - k + off))
                slope = s
        return terms, (f(lo) if gate is None else 0)

    def pwl(self, z: Expr, lo: int, hi: int, f, name: str, lits=None) -> Expr:
        self.ranges.append((z, lo, hi, name))
        terms, const = self.pwl_terms(z, lo, hi, f, lits)
        if not terms:
            return self.one * const
        return self.ffn(terms, name) + const

    def pwl_cases(self, z: Expr, lo: int, hi: int, cases, name: str) -> Expr:
        """sum over (lits, f) of f(z) gated by lits, in one FFN node."""
        self.ranges.append((z, lo, hi, name))
        terms, const = [], 0
        for lits, f in cases:
            t, k = self.pwl_terms(z, lo, hi, f, lits)
            terms += t
            const += k
        return (self.ffn(terms, name) if terms else Expr()) + const

    def step(self, x: Expr, name: str = "step") -> Expr:
        """[x >= 0] for any integer x."""
        return self.ffn([(1, self.one, x + 1), (-1, self.one, x)], name)

    def eq(self, x: Expr, k=0, name: str = "eq") -> Expr:
        d = x - k
        return self.ffn([(1, self.one, d + 1), (-2, self.one, d), (1, self.one, d - 1)], name)


def tokexpr(fam: str, payload) -> Expr:
    return as_expr(payload) + base(fam)


def reg_tok(fam, val, c, v, px, ln) -> Expr:
    return base(fam) + val + 256 * c + 512 * v + 1024 * px + 2048 * ln


def p_tok(C, Z, I, D, V, N, px, ln) -> Expr:
    return base("P") + C + 2 * Z + 4 * I + 8 * D + 16 * V + 32 * N + 64 * px + 128 * ln


# --- PRG banking: each board's states, their literals, and the PRG index of a CPU address -----

def board_states(c: NesCircuit, tf, is_map):
    """Latest heads for the board's CPU-visible state. Returns (states, extra) where states is a
    list of (literals, index_fn(a) -> PRG offset) and extra holds dims the circuit needs."""
    cfg = c.cfg
    n16 = cfg.n16
    extra = {}
    if cfg.board in (0, 3):  # fixed PRG
        return [([], lambda a: (a - 0x8000) % cfg.n_prg)], extra
    if cfg.board == 2:  # UxROM: switchable 16 KB at $8000, last bank at $C000
        onehot = [tf(f"uxrom.bank{b}", lambda fm, p, b=b: int(fm == "MAP" and (p & 255)
                                                              & (n16 - 1) == b))
                  for b in range(n16)]
        val = tf("uxrom.val", lambda fm, p: (p & 255) if fm == "MAP" else 0)
        got = c.latest(is_map, [val, *onehot], "uxrom")
        extra["map_dims"] = {"bank": got[0]}

        def idx(b):
            return lambda a: (b if a < 0xC000 else n16 - 1) * BANK + (a & 0x3FFF)
        return [([got[1 + b]], idx(b)) for b in range(n16)], extra
    assert cfg.board == 1  # MMC1

    def commit(reg):
        return lambda fm, p: int(fm == "MAP" and p >> 8 & 1 and (p >> 9 & 3) == reg)

    is_ctrl = tf("mmc1.ctrl_commit", commit(0))
    is_prg = tf("mmc1.prg_commit", commit(3))
    v5 = lambda fm, p: (p & 31) if fm == "MAP" else 0
    ctrl_val = tf("mmc1.val5", v5)
    modes = [tf(f"mmc1.mode{m}", lambda fm, p, m=m: int(fm == "MAP" and (p >> 2 & 3) == m))
             for m in range(4)]
    cb2 = tf("mmc1.v.b2", lambda fm, p: (p >> 2 & 1) if fm == "MAP" else 0)
    cb3 = tf("mmc1.v.b3", lambda fm, p: (p >> 3 & 1) if fm == "MAP" else 0)
    prgs = [tf(f"mmc1.prg{b}", lambda fm, p, b=b: int(fm == "MAP" and (p & 15) & (n16 - 1) == b))
            for b in range(n16)]
    sh = tf("mmc1.shift", lambda fm, p: (p & 31) if fm == "MAP" and not p >> 8 & 1 else 0)
    cnt = tf("mmc1.count", lambda fm, p: (p >> 5 & 7) if fm == "MAP" and not p >> 8 & 1 else 0)
    cnts = [tf(f"mmc1.count{k}", lambda fm, p, k=k: int(fm == "MAP" and (0 if p >> 8 & 1 else
                                                                         p >> 5 & 7) == k))
            for k in range(5)]
    ctrl, *rest = c.latest(is_ctrl, [ctrl_val, *modes, cb2, cb3], "mmc1.ctrl")
    MODE, c2, c3 = rest[:4], rest[4], rest[5]
    prg, *PRG = c.latest(is_prg, [ctrl_val, *prgs], "mmc1.prg")
    shift, count, *CNT = c.latest(is_map, [sh, cnt, *cnts], "mmc1.shift")
    extra["map_dims"] = {"ctrl": ctrl, "prg": prg, "shift": shift, "count": count}
    extra["mmc1"] = dict(ctrl=ctrl, c2=c2, c3=c3, shift=shift, CNT=CNT)

    def idx(m, p):
        def f(a):
            if m < 2:
                return (p & ~1) * BANK + (a & 0x7FFF)
            if m == 2:
                return (0 if a < 0xC000 else p) * BANK + (a & 0x3FFF)
            return (p if a < 0xC000 else n16 - 1) * BANK + (a & 0x3FFF)
        return f
    return [([MODE[m], PRG[p]], idx(m, p)) for m in range(4) for p in range(n16)], extra


def region_banks(c: NesCircuit, states):
    """Per 16 KB region ($8000, $C000) and bank: a literal "this bank is mapped there" and the
    ROM-head position of an address a ($8000-$FFFF) when it is (else 0). Each bank's literal is
    one FFN node over the board states that map it (layer 1), so the ROM queries need one
    gated function per bank instead of one per board state."""
    rom_pos = c.cfg.layout.rom_pos
    out = []
    for r, lo in enumerate((0x8000, 0xC000)):
        by_bank: dict[int, list] = {}
        for lits, ix in states:
            b = ix(lo) // BANK
            for a in (lo, lo + 0x1234, lo + 0x3FFF):
                assert ix(a) == b * BANK + (a & 0x3FFF), "PRG index must be bank * 16K + offset"
            by_bank.setdefault(b, []).append(lits)
        for b, group in sorted(by_bank.items()):
            if group == [[]]:
                lit = []  # a fixed bank
            elif len(group) == 1 and len(group[0]) == 1:
                lit = group[0]
            else:
                lit = [c.ffn([(1, c.one, c.AND(g)) for g in group], f"bank{r}={b}")]

            def g(a, r=r, b=b):
                return rom_pos + b * BANK + (a & 0x3FFF) if (a >= 0xC000) == bool(r) else 0
            out.append((lit, g))
    return out


def build(cfg: Config = Config()) -> NesCircuit:
    def ram_code(a: int) -> bool:  # an address below $8000 where code may run
        return a < 0x2000 or (cfg.prgram and 0x6000 <= a < 0x8000)

    c = NesCircuit(cfg)
    lay = cfg.layout
    one, pos, pos2 = c.one, c.pos, c.pos2
    tf = c.tf
    mapper_board = cfg.board != 0

    # =========================================================================== embedding
    def isf(*fams):
        fs = set(fams)
        return lambda fm, p: int(fm in fs)

    f_fin = tf("final", vfinal)
    f_len = tf("len", vlen)
    is_anc = tf("is_anchor", isf("J", "JI", "R", "JL", "RL"))
    f_anc = tf("anchor", lambda fm, p: S.anchor_addr(fm, p) if fm in ("J", "JI", "R", "JL", "RL")
               else 0)
    f_anc_lo = tf("anchor.lo", lambda fm, p: S.anchor_addr(fm, p) & 255
                  if fm in ("J", "JI", "R", "JL", "RL") else 0)
    f_anc_hi = tf("anchor.hi", lambda fm, p: S.anchor_addr(fm, p) >> 8
                  if fm in ("J", "JI", "R", "JL", "RL") else 0)
    is_ = {fm: tf(f"is_{fm}", isf(fm)) for fm in
           ["A", "X", "Y", "LO", "HI", "IN", "IOR", "IOW", "DM", "PSH", "PUL", "P", "ST", "RM",
            "NMI", "GO", "RB", "RI", "MAP", "OPF", "OB1", "OB2"]}

    def byte_of(*fams):
        fs = set(fams)
        return lambda fm, p: (p & 255) if fm in fs else 0

    def bit_of(fam, i):
        return lambda fm, p: (p >> i & 1) if fm == fam else 0

    vA = tf("A.val", byte_of("A"))
    vA_bits = [tf(f"A.b{i}", bit_of("A", i)) for i in range(8)]
    vX, vX0, vX7 = tf("X.val", byte_of("X")), tf("X.b0", bit_of("X", 0)), tf("X.b7", bit_of("X", 7))
    vY, vY0, vY7 = tf("Y.val", byte_of("Y")), tf("Y.b0", bit_of("Y", 0)), tf("Y.b7", bit_of("Y", 7))
    nzc_src = tf("nzc_src", isf("A", "X", "Y", "P", "RM"))

    def flag_n(fm, p):
        return (p >> 5 & 1) if fm == "P" else (p >> 7 & 1) if fm in ("A", "X", "Y", "RM") else 0

    def flag_z(fm, p):
        return (p >> 1 & 1) if fm == "P" else int(p & 255 == 0) if fm in ("A", "X", "Y", "RM") \
            else 0

    def flag_c(fm, p):
        return (p & 1) if fm == "P" else (p >> 8 & 1) if fm in ("A", "X", "Y", "RM") else 0

    fN, fZ, fC = tf("fN", flag_n), tf("fZ", flag_z), tf("fC", flag_c)
    v_src = tf("v_src", isf("A", "X", "Y", "P"))
    fV = tf("fV", lambda fm, p: (p >> 4 & 1) if fm == "P" else (p >> 9 & 1)
            if fm in ("A", "X", "Y") else 0)
    i_src = tf("i_src", isf("P", "JI"))
    fI = tf("fI", lambda fm, p: (p >> 2 & 1) if fm == "P" else int(fm == "JI"))
    fD = tf("fD", lambda fm, p: (p >> 3 & 1) if fm == "P" else 0)
    sp_src = tf("sp_src", isf("S", "PSH", "PUL"))
    fSP = tf("fSP", byte_of("S", "PSH", "PUL"))
    lo_val = tf("LO.val", byte_of("LO"))
    lo_b7 = tf("LO.b7", bit_of("LO", 7))
    lo_l3 = tf("LO.l3", lambda fm, p: (p & 7) if fm == "LO" else 0)
    lo_l5 = tf("LO.l5", lambda fm, p: (p & 31) if fm == "LO" else 0)
    hi_val = tf("HI.val", byte_of("HI"))
    in_val = tf("IN.val", byte_of("IN"))
    iow_val = tf("IOW.val", byte_of("IOW"))
    is_dma = tf("is_dma_start", lambda fm, p: int(fm == "IOW" and (p >> 8 & 63) == 0x14))
    dm_idx = tf("DM.idx", lambda fm, p: (p >> 8) if fm == "DM" else 0)
    dm_last = tf("DM.last", lambda fm, p: int(fm == "DM" and p >> 8 == 255))
    st_val = tf("ST.val", byte_of("ST"))
    rm_val = tf("RM.val", byte_of("RM"))
    rb_val = tf("RB.val", byte_of("RB"))
    ri_val = tf("RI.val", byte_of("RI"))
    rb_b7 = tf("RB.b7", bit_of("RB", 7))
    rb_l3 = tf("RB.l3", lambda fm, p: (p & 7) if fm == "RB" else 0)
    rb_l5 = tf("RB.l5", lambda fm, p: (p & 31) if fm == "RB" else 0)
    # a byte's meaning as an opcode: ROM bytes (fetched by position) and OPF tokens (code in
    # RAM, read into the trace by the model itself)
    def opc(fm, p):
        return p if fm == "RB" else (p & 255) if fm == "OPF" else None

    op_valid = tf("op.valid", lambda fm, p: int(opc(fm, p) in OPS))
    op_len = tf("op.len", lambda fm, p: LENGTH[OPS[opc(fm, p)][1]] if opc(fm, p) in OPS else 0)
    op_mn = {mn: tf(f"op.{mn}", lambda fm, p, mn=mn: int(opc(fm, p) in OPS and
                                                          OPS[opc(fm, p)][0] == mn)) for mn in MNS}
    op_mode = {m: tf(f"mode.{m}", lambda fm, p, m=m: int(opc(fm, p) in OPS and
                                                          OPS[opc(fm, p)][1] == m)) for m in MODES}
    ob1_val, ob1_b7 = tf("OB1.val", byte_of("OB1")), tf("OB1.b7", bit_of("OB1", 7))
    ob1_l3 = tf("OB1.l3", lambda fm, p: (p & 7) if fm == "OB1" else 0)
    ob1_l5 = tf("OB1.l5", lambda fm, p: (p & 31) if fm == "OB1" else 0)
    ob2_val, ob2_b7 = tf("OB2.val", byte_of("OB2")), tf("OB2.b7", bit_of("OB2", 7))
    is_fetch = is_["OPF"] + is_["OB1"] + is_["OB2"]
    f_nf = tf("fetch.n", lambda fm, p: {"OPF": 1, "OB1": 2, "OB2": 3}.get(fm, 0))
    f_last = tf("fetch.last", lambda fm, p: int(fm == "OB2" or fm in ("OPF", "OB1") and p >> 8))
    is_entry = sum((is_[f] for f in ("RI", "IN", "ST", "RM", "PSH")), Expr())

    # =========================================================================== L1 attn
    (B,) = c.cumsum([f_len], "bytes")  # instruction bytes committed so far
    (fpos,) = c.latest(f_fin, [pos], "rec_start")
    A, *A_bits = c.latest(is_["A"], [vA, *vA_bits], "A")
    X, X0, X7 = c.latest(is_["X"], [vX, vX0, vX7], "X")
    Y, Y0, Y7 = c.latest(is_["Y"], [vY, vY0, vY7], "Y")
    N, Z, C = c.latest(nzc_src, [fN, fZ, fC], "NZC")
    (V,) = c.latest(v_src, [fV], "V")
    (I,) = c.latest(i_src, [fI], "I")
    (D,) = c.latest(is_["P"], [fD], "D")
    (SP,) = c.latest(sp_src, [fSP], "SP")
    LO, LO7, LO3, LO5 = c.latest(is_["LO"], [lo_val, lo_b7, lo_l3, lo_l5], "LO")
    (HI,) = c.latest(is_["HI"], [hi_val], "HI")
    (PAGE,) = c.latest(is_["IOW"], [iow_val], "IOW")
    nmi_pos, nmi_found = c.latest(is_["NMI"], [pos, is_["NMI"]], "NMI")
    (started,) = c.latest(is_["GO"], [is_["GO"]], "GO")
    # instruction bytes of code in RAM (zeros in a record that has none)
    lat_op = c.latest(is_["OPF"] + f_fin, [op_valid, op_len, *op_mn.values(), *op_mode.values()],
                      "latch.op")
    lat_b1 = c.latest(is_["OB1"] + f_fin, [ob1_val, ob1_b7, ob1_l3, ob1_l5], "latch.b1")
    lat_b2 = c.latest(is_["OB2"] + f_fin, [ob2_val, ob2_b7], "latch.b2")
    (nf,) = c.latest(is_fetch + f_fin, [f_nf], "latch.n")
    states, extra = board_states(c, tf, is_["MAP"])
    c.map_dims = extra.get("map_dims", {})
    banks = region_banks(c, states)

    # =========================================================================== L1 ffn
    j = pos - fpos - nf  # tokens of the current record so far (after any instruction fetch)
    c.ranges.append((j, 0, 300, "j"))
    J1, J3, J5 = (c.eq(j, k, f"j=={k}") for k in (1, 3, 5))
    NM = c.step(nmi_pos - fpos - 1 - 2 * MAX_POS * (1 - nmi_found), "in_nmi")
    jj = j - NM  # stage within BRK / NMI / JSR push sequences
    spm = c.pwl(SP, 0, 255, lambda v: (v - 1) & 255, "SP-1")
    spp = c.pwl(SP, 0, 255, lambda v: (v + 1) & 255, "SP+1")
    ramkey = c.ffn([(1, pos - lay.ram_pos, is_["RI"])], "ramkey")
    ramkey2 = c.ffn([(1, pos2 - 2 * lay.ram_pos * pos + lay.ram_pos ** 2, is_["RI"])], "ramkey^2")
    pbyte = C + 2 * Z + 4 * I + 8 * D + 32 + 64 * V + 128 * N  # P with B = 0
    lo_y = LO + Y  # (zp),Y: pointer bytes are latched in LO / HI
    iny_ea = c.pwl(lo_y + 256 * HI, 0, 0xFFFF + 255, lambda v: v & 0xFFFF, "iny.ea")
    iny_px = c.pwl(lo_y, 0, 510, lambda v: int(v > 255), "iny.px")
    iny_a7 = c.pwl(lo_y, 0, 510, lambda v: v >> 7 & 1, "iny.a7")
    iny_l3 = c.pwl(lo_y, 0, 510, lambda v: v & 7, "iny.l3")
    iny_l5 = c.pwl(lo_y, 0, 510, lambda v: v & 31, "iny.l5")
    inx_ea = LO + 256 * HI
    incX = c.pwl(X, 0, 255, lambda v: (v + 1) & 255, "X+1")
    decX = c.pwl(X, 0, 255, lambda v: (v - 1) & 255, "X-1")
    incY = c.pwl(Y, 0, 255, lambda v: (v + 1) & 255, "Y+1")
    decY = c.pwl(Y, 0, 255, lambda v: (v - 1) & 255, "Y-1")

    # =========================================================================== L2
    anc, Ba, anc_lo, anc_hi = c.latest(is_anc, [f_anc, B, f_anc_lo, f_anc_hi], "anchor")
    off = B - Ba
    u = anc + off  # the PC, before wrapping at 16 bits
    UH = 0xFFFF + 64
    pc = c.pwl(u, 0, UH, lambda v: v & 0xFFFF, "pc")
    pc1 = c.pwl(u, 0, UH, lambda v: (v + 1) & 0xFFFF, "pc+1")
    pc2 = c.pwl(u, 0, UH, lambda v: (v + 2) & 0xFFFF, "pc+2")
    pc_ram = c.pwl(u, 0, UH, lambda v: int(v & 0xFFFF < 0x8000), "pc<8000")

    # ROM-head queries for PC, PC + 1, PC + 2, one gated function per (region, bank)
    # (0: code in RAM)
    def fetch_f(k, g):
        return lambda v: 0 if v & 0xFFFF < 0x8000 else g(0x8000 | ((v + k) & 0x7FFF))

    fetch_q = [c.pwl_cases(u, 0, UH, [(lits, fetch_f(k, g)) for lits, g in banks],
                           f"fetch.q{k}") for k in range(3)]
    REF = c.pwl(off, 0, 200, lambda v: int(v >= S.OFF_MAX), "refresh")
    JJ1, JJ2, JJ3 = (c.eq(jj, k, f"jj=={k}") for k in (1, 2, 3))

    # =========================================================================== L3 attn
    def rom_read(q, values, name):
        return c.lookup(q, 2 * pos, -pos2, values, (0, cfg.n_prg), name)

    mn_names, mode_names = list(op_mn), list(op_mode)
    got = rom_read(fetch_q[0], [op_valid, op_len, *op_mn.values(), *op_mode.values()], "fetch.op")
    got = [g + lt for g, lt in zip(got, lat_op)]  # ROM code or RAM code, never both
    valid, LEN = got[0], got[1]
    MN = dict(zip(mn_names, got[2:2 + len(mn_names)]))
    MODE = dict(zip(mode_names, got[2 + len(mn_names):]))
    b1, b1_7, b1_l3, b1_l5 = (g + lt for g, lt in zip(
        rom_read(fetch_q[1], [rb_val, rb_b7, rb_l3, rb_l5], "fetch.b1"), lat_b1))
    b2, b2_7 = (g + lt for g, lt in zip(rom_read(fetch_q[2], [rb_val, rb_b7], "fetch.b2"),
                                         lat_b2))

    def mns(*names) -> Expr:
        return sum((MN[n] for n in names), Expr())

    def modes(*names) -> Expr:
        return sum((MODE[n] for n in names), Expr())

    # =========================================================================== L3 ffn
    fl_of = {"N": N, "V": V, "C": C, "Z": Z}
    taken = c.ffn([(1, one, c.AND([MN[mn], fl_of[fl] if want else 1 - fl_of[fl]]))
                   for mn, (fl, want) in BRANCH_FLAG.items()], "taken")
    absa = b1 + 256 * b2
    sx, sy = b1 + X, b1 + Y
    zpx = c.pwl(sx, 0, 510, lambda v: v & 255, "zpx")
    zpy = c.pwl(sy, 0, 510, lambda v: v & 255, "zpy")
    zpx1 = c.pwl(sx, 0, 510, lambda v: (v + 1) & 255, "zpx+1")
    b1p1 = c.pwl(b1, 0, 255, lambda v: (v + 1) & 255, "b1+1")
    x_a7 = c.pwl(sx, 0, 510, lambda v: v >> 7 & 1, "x.a7")
    y_a7 = c.pwl(sy, 0, 510, lambda v: v >> 7 & 1, "y.a7")
    x_l3 = c.pwl(sx, 0, 510, lambda v: v & 7, "x.l3")
    y_l3 = c.pwl(sy, 0, 510, lambda v: v & 7, "y.l3")
    x_l5 = c.pwl(sx, 0, 510, lambda v: v & 31, "x.l5")
    y_l5 = c.pwl(sy, 0, 510, lambda v: v & 31, "y.l5")
    x_px = c.pwl(sx, 0, 510, lambda v: int(v > 255), "x.px")
    y_px = c.pwl(sy, 0, 510, lambda v: int(v > 255), "y.px")
    abx = c.pwl(absa + X, 0, 0xFFFF + 255, lambda v: v & 0xFFFF, "abx")
    aby = c.pwl(absa + Y, 0, 0xFFFF + 255, lambda v: v & 0xFFFF, "aby")
    ptr2 = absa + 1 - 256 * c.eq(b1, 255, "b1==255")  # JMP ($xxFF) wraps within the page
    jr = pc + 2 + b1 - 256 * b1_7  # branch target, before wrapping
    jt = c.pwl(jr, -128, 0xFFFF + 130, lambda v: v & 0xFFFF, "jt")
    jt_rom = c.pwl(jr, -128, 0xFFFF + 130, lambda v: int(v & 0xFFFF >= 0x8000), "jt.rom")
    jt_ram = c.pwl(jr, -128, 0xFFFF + 130, lambda v: int(ram_code(v & 0xFFFF)), "jt.ram")
    abs_ram = c.pwl(b2, 0, 255, lambda v: int(ram_code(v << 8)), "abs.ram")  # code in RAM ok
    # return address bytes (pc + 2, or pc for NMI), from the anchor's bytes: the low byte plus
    # the small offset carries at most once into the high byte
    rs = anc_lo + off + 2 - 2 * NM
    pcl = c.pwl(rs, 0, 255 + 66, lambda v: v & 255, "pcl")
    pch = anc_hi + c.step(rs - 256, "pcl.carry") - 256 * c.step(rs - 256 + 512 * (anc_hi - 255),
                                                                   "pch.wrap")
    xs = c.ffn([(1, one, c.AND([f_fin, 1 - REF, 1 - pc_ram])), (1, one, f_last)], "exec_start")
    fetch_last = c.pwl(LEN, 0, 3, lambda v: int(v == 2), "fetch.len2")
    sval = c.sel([([MN["STA"]], A), ([MN["STX"]], X), ([MN["STY"]], Y)], "store.val")
    sv0 = c.sel([([MN["STA"]], A_bits[0]), ([MN["STX"]], X0), ([MN["STY"]], Y0)], "store.b0")
    sv7 = c.sel([([MN["STA"]], A_bits[7]), ([MN["STX"]], X7), ([MN["STY"]], Y7)], "store.b7")
    rcmp = c.sel([([MN["CMP"]], A), ([MN["CPX"]], X), ([MN["CPY"]], Y)], "cmp.reg")

    # =========================================================================== L4 ffn
    S0 = [xs]  # the instruction executes (its bytes are known): the record's first token, or
    # the first after the fetch tokens of code in RAM
    nNM = 1 - NM
    direct = modes(*DIRECT)
    ind_y = modes("inx", "iny")
    cand = {"imm": pc1, "zp": b1, "zpx": zpx, "zpy": zpy, "abs": absa, "abx": abx,
            "aby": aby}
    qa_cases = []
    for m, a in cand.items():
        qa_cases.append(([MODE[m], *S0], a))
        qa_cases.append(([MODE[m], is_["IN"], nNM], a))
    rm_mid = is_["RM"] - f_fin  # an RM token that is not final: RMW on mapper registers
    for m in ("zp", "zpx", "abs", "abx"):
        qa_cases.append(([MODE[m], rm_mid, nNM], cand[m]))
    fetching = [f_fin, 1 - REF, pc_ram]  # code in RAM: read the opcode into the trace
    qa_cases += [
        (fetching, pc), ([is_["OPF"], 1 - f_last], pc1), ([is_["OB1"], 1 - f_last], pc2),
        ([MODE["inx"], *S0], zpx), ([MODE["inx"], is_["LO"], nNM], zpx1),
        ([MODE["inx"], is_["HI"] + is_["IN"], nNM], inx_ea),
        ([MODE["iny"], *S0], b1), ([MODE["iny"], is_["LO"], nNM], b1p1),
        ([MODE["iny"], is_["HI"] + is_["IN"], nNM], iny_ea),
        ([MODE["ind"], *S0], absa), ([MODE["ind"], is_["LO"], nNM], ptr2),
        ([mns("RTS", "RTI", "PLA", "PLP"), is_["PUL"], nNM], 0x100 + SP),
        ([MN["BRK"], is_["PSH"], JJ3, nNM], 0xFFFE), ([MN["BRK"], is_["LO"], nNM], 0xFFFF),
        ([NM, is_["PSH"], JJ3], 0xFFFA), ([NM, is_["LO"]], 0xFFFB),
        ([is_dma + is_["DM"]], 256 * PAGE + dm_idx + is_["DM"]),
    ]
    qa = c.sel(qa_cases, "qa")
    c.ranges.append((qa, 0, 0xFFFF, "qa"))

    def by_mode(vals, name):
        return c.sel([([MODE[m]], v) for m, v in vals.items()], name)

    a7 = by_mode({"zp": b1_7, "zpx": x_a7, "zpy": y_a7, "abs": b1_7, "abx": x_a7, "aby": y_a7,
                  "inx": LO7, "iny": iny_a7}, "ea.a7")
    l3 = by_mode({"abs": b1_l3, "abx": x_l3, "aby": y_l3, "inx": LO3, "iny": iny_l3}, "ea.l3")
    l5 = by_mode({"abs": b1_l5, "abx": x_l5, "aby": y_l5, "inx": LO5, "iny": iny_l5}, "ea.l5")
    pp = mns(*PAGE_PENALTY)
    px = c.sel([([MODE["abx"], pp], x_px), ([MODE["aby"], pp], y_px),
                ([MODE["iny"], pp], iny_px)], "px")
    # the byte the next token pushes, and where (0x100 + SP), when it is a PSH
    psh_cases = [
        ([MN["JSR"], b2_7 + abs_ram, *S0], pch), ([MN["JSR"], is_["PSH"], JJ1, nNM], pcl),
        ([MN["BRK"], *S0], pch), ([MN["BRK"], is_["PSH"], JJ1, nNM], pcl),
        ([MN["BRK"], is_["PSH"], JJ2, nNM], pbyte + 16),
        ([is_["NMI"]], pch), ([NM, is_["PSH"], JJ1], pcl), ([NM, is_["PSH"], JJ2], pbyte),
        ([MN["PHA"], *S0], A), ([MN["PHP"], *S0], pbyte + 16),
    ]
    psh_data = c.sel([([started, *lits], v) for lits, v in psh_cases], "push.data")

    # =========================================================================== L5 ffn
    def ram_key(a: int) -> int:
        if a < 0x2000:
            return a & 0x7FF
        if a < cfg.io_end:
            return T.K_IO
        if a < 0x8000:
            return T.K_PRAM + a - 0x6000
        return T.K_ZERO

    def wram(a: int) -> int:  # where a store to a lands in the RAM head (K_NULL: nowhere)
        return ram_key(a) if a < 0x2000 or cfg.io_end <= a < 0x8000 else cfg.k_null

    q_ram = c.pwl(qa, 0, 0xFFFF, ram_key, "ram.q")
    q_rom = c.pwl_cases(qa, 0, 0xFFFF, [(lits, lambda a, g=g: 0 if a < 0x8000 else g(a))
                                        for lits, g in banks], "rom.q")
    ge = {b: c.pwl(qa, 0, 0xFFFF, lambda v, b=b: int(v >= b), f"qa>={b:X}")
          for b in (0x2000, 0x4000, 0x4020, cfg.io_end, 0x8000)}
    r_io = ge[0x2000] - ge[cfg.io_end]
    r_rom = ge[0x8000]
    r_ppu, r_apu, r_oth = ge[0x2000] - ge[0x4000], ge[0x4000] - ge[0x4020], ge[0x4020] - \
        ge[cfg.io_end]
    is4014 = c.eq(qa, 0x4014, "qa==4014")
    stores, rmw = mns(*STORES), mns(*RMWS)
    w_lits = [[started, *lits] for lits in ([stores, direct, *S0], [stores, ind_y, is_["HI"], nNM],
                                            [rmw, direct, *S0], [rmw, is_["IN"], nNM])]
    # the write key of the next token, as high and low bytes (PSH: 0x100 + SP)
    psh_lits = [[started, *lits] for lits, _ in psh_cases]
    wh = c.pwl_cases(qa, 0, 0xFFFF, [(lits, lambda a: wram(a) >> 8) for lits in w_lits], "wkey.h")
    wh = wh + c.sel([(lits, 1) for lits in psh_lits], "wkey.psh")
    wl = c.pwl_cases(qa, 0, 0xFFFF, [(lits, lambda a: wram(a) & 255) for lits in w_lits],
                     "wkey.l")
    wl = wl + c.sel([(lits, SP) for lits in psh_lits], "wkey.sp")
    if cfg.board == 1:
        mreg = c.pwl(qa, 0, 0xFFFF, lambda a: a >> 13 & 3, "mmc1.reg")

    # =========================================================================== L6
    wh_p, wl_p, psh_p = c.prev([wh, wl, psh_data], "prev.write")
    kh = cfg.k_null >> 8
    sq_h = c.pwl(wh_p, 0, kh, lambda v: v * v, "wkey.h^2")
    sq_l = c.pwl(wl_p, 0, 255, lambda v: v * v, "wkey.l^2")
    hl = c.ffn([(1, wh_p, wl_p)], "wkey.h*l")
    ioreg = c.ffn([(1, 32 + l3, r_ppu), (1, l5, r_apu), (40, one, r_oth)], "io.reg")

    # =========================================================================== L7 attn
    tq = MAX_POS
    key = ramkey + T.K_IO * is_["IN"] + 256 * wh_p + wl_p
    key2 = ramkey2 + T.K_IO ** 2 * is_["IN"] + 65536 * sq_h + 512 * hl + sq_l
    mval = ri_val + in_val + st_val + rm_val + psh_p
    # the second value (the writer's position) is unused by the CPU; it answers "who last wrote
    # this byte?" for tools (see Session.writer)
    M_ram, _ = c.lookup(q_ram, 2 * tq * key, -tq * key2 + pos + 2 * MAX_POS * (is_entry - 1),
                        [mval, pos], (0, cfg.k_null), "RAM")
    (M_rom,) = rom_read(q_rom, [rb_val], "ROM")
    M = M_ram + M_rom

    # =========================================================================== L7 ffn
    Mb = [c.pwl(M, 0, 255, lambda v, i=i: v >> i & 1, f"M.b{i}") for i in range(8)]
    s_adc = A + M + C
    s_sbc = A + 255 - M + C
    adc_r = c.pwl(s_adc, 0, 511, lambda v: v & 255, "adc.r")
    adc_c = c.pwl(s_adc, 0, 511, lambda v: int(v > 255), "adc.c")
    adc_7 = c.pwl(s_adc, 0, 511, lambda v: v >> 7 & 1, "adc.7")
    sbc_r = c.pwl(s_sbc, 0, 511, lambda v: v & 255, "sbc.r")
    sbc_c = c.pwl(s_sbc, 0, 511, lambda v: int(v > 255), "sbc.c")
    sbc_7 = c.pwl(s_sbc, 0, 511, lambda v: v >> 7 & 1, "sbc.7")
    dcmp = rcmp - M + 256
    cmp_c = c.pwl(dcmp, 1, 511, lambda v: int(v >= 256), "cmp.c")
    cmp_z = c.pwl(dcmp, 1, 511, lambda v: int(v == 256), "cmp.z")
    cmp_n = c.pwl(dcmp, 1, 511, lambda v: v >> 7 & 1, "cmp.n")
    incM = c.pwl(M, 0, 255, lambda v: (v + 1) & 255, "M+1")
    decM = c.pwl(M, 0, 255, lambda v: (v - 1) & 255, "M-1")
    rts_t = c.pwl(LO + 256 * M + 1, 1, 0x10000, lambda v: v & 0xFFFF, "rts.t")
    rts_rom = c.pwl(LO + 256 * M + 1, 1, 0x10000, lambda v: int(v & 0xFFFF >= 0x8000), "rts.rom")
    rts_ram = c.pwl(LO + 256 * M + 1, 1, 0x10000, lambda v: int(ram_code(v & 0xFFFF)), "rts.ram")
    ind_t = LO + 256 * M  # JMP (ind) / RTI target
    ind_ram = c.pwl(ind_t, 0, 0xFFFF, lambda v: int(ram_code(v)), "ind.ram")
    op_last = c.pwl(M, 0, 255, lambda v: int(v not in OPS or LENGTH[OPS[v][1]] == 1), "op.last")

    # =========================================================================== L8 ffn
    M7, M6 = Mb[7], Mb[6]
    and_ = c.ffn([(1 << i, A_bits[i], Mb[i]) for i in range(8)], "A&M")
    A7 = A_bits[7]
    adc_v = c.ffn([(1, one, A7 + M7 - adc_7 - 1), (1, one, adc_7 - A7 - M7)], "adc.v")
    sbc_v = c.ffn([(1, one, A7 - M7 - sbc_7), (1, one, M7 + sbc_7 - A7 - 1)], "sbc.v")
    common = c.ffn([(1, one, A_bits[i] + Mb[i] - 1) for i in range(8)], "A&M.count")
    Mz = c.ffn([(1, M, 1 - r_io)], "dma.byte")  # an I/O page reads as 0

    # =========================================================================== L9 ffn
    bit_z = c.eq(common, 0, "bit.z")

    # =========================================================================== L10: output
    cases = []

    def case(lits, tok):
        cases.append((list(lits), tok))

    ora = A + M - and_
    eor = A + M - 2 * and_
    lsrM = sum((Mb[i] * (1 << (i - 1)) for i in range(1, 8)), Expr())
    lsrA = sum((A_bits[i] * (1 << (i - 1)) for i in range(1, 8)), Expr())
    read_final = {
        "LDA": reg_tok("A", M, C, V, px, LEN), "LDX": reg_tok("X", M, C, V, px, LEN),
        "LDY": reg_tok("Y", M, C, V, px, LEN),
        "ADC": reg_tok("A", adc_r, adc_c, adc_v, px, LEN),
        "SBC": reg_tok("A", sbc_r, sbc_c, sbc_v, px, LEN),
        "AND": reg_tok("A", and_, C, V, px, LEN), "ORA": reg_tok("A", ora, C, V, px, LEN),
        "EOR": reg_tok("A", eor, C, V, px, LEN),
        "CMP": p_tok(cmp_c, cmp_z, I, D, V, cmp_n, px, LEN),
        "CPX": p_tok(cmp_c, cmp_z, I, D, V, cmp_n, px, LEN),
        "CPY": p_tok(cmp_c, cmp_z, I, D, V, cmp_n, px, LEN),
        "BIT": p_tok(C, bit_z, I, D, M6, M7, 0, LEN),
    }
    rmw_res = {  # (result, carry)
        "ASL": (2 * M - 256 * M7, M7), "LSR": (lsrM, Mb[0]),
        "ROL": (2 * M - 256 * M7 + C, M7), "ROR": (lsrM + 128 * C, Mb[0]),
        "INC": (incM, C), "DEC": (decM, C),
    }
    def mmc1_write(lits, b0, b7):
        """The MAP token of a write of a byte with bits b0, b7 to MMC1's serial port."""
        m1 = extra["mmc1"]
        case([*lits, b7], base("MAP") + m1["ctrl"] + 4 * (1 - m1["c2"]) + 8 * (1 - m1["c3"])
             + 256 + 2048 * LEN)
        for k in range(4):
            case([*lits, 1 - b7, m1["CNT"][k]],
                 base("MAP") + m1["shift"] + (1 << k) * b0 + 32 * (k + 1) + 2048 * LEN)
        case([*lits, 1 - b7, m1["CNT"][4]],
             base("MAP") + m1["shift"] + 16 * b0 + 256 + 512 * mreg + 2048 * LEN)

    final_stage = [[direct, *S0, 1 - r_io], [is_["IN"], nNM], [ind_y, is_["HI"], nNM, 1 - r_io]]
    ior_stage = [[direct, *S0, r_io], [ind_y, is_["HI"], nNM, r_io]]
    reads = mns(*READS)
    ior = tokexpr("IOR", ioreg)

    case([f_fin, REF, 1 - pc_ram], tokexpr("R", pc - 0x8000))
    case([f_fin, REF, pc_ram], tokexpr("RL", pc))
    # code in RAM: its bytes go into the trace before it runs
    case(fetching, tokexpr("OPF", M + 256 * op_last))
    case([is_["OPF"], 1 - f_last], tokexpr("OB1", M + 256 * fetch_last))
    case([is_["OB1"], 1 - f_last], tokexpr("OB2", M))
    case([*S0, 1 - valid], S.tok("HALT"))
    # loads, ALU, compares, BIT
    for mn, tok in read_final.items():
        for st in final_stage:
            case([MN[mn], *st], tok)
    for st in ior_stage:
        case([reads, *st], ior)
    case([reads + stores, ind_y, *S0], tokexpr("LO", M))
    case([reads + stores, ind_y, is_["LO"], nNM], tokexpr("HI", M))
    # stores
    st_mem = 1 - r_io - (r_rom if mapper_board else 0)  # RAM, PRG-RAM (and ROM on NROM)
    for st in ([direct, *S0], [ind_y, is_["HI"], nNM]):
        case([stores, *st, is4014], base("IOW") + sval + 0x14 * 256)
        case([stores, *st, r_io, 1 - is4014], base("IOW") + sval + 256 * ioreg + 16384 * LEN)
        case([stores, *st, st_mem], base("ST") + sval + 256 * a7 + 512 * LEN)
        if cfg.board in (2, 3):
            case([stores, *st, r_rom], base("MAP") + sval + 2048 * LEN)
        elif cfg.board == 1:
            mmc1_write([stores, *st, r_rom], sv0, sv7)
    case([is_dma], base("DM") + Mz)
    case([is_["DM"], 1 - dm_last], base("DM") + Mz + 256 * (dm_idx + 1))
    case([is_["DM"], dm_last], tokexpr("NT", LEN))
    # read-modify-write
    acc = {"ASL": (2 * A - 256 * A7, A7), "LSR": (lsrA, A_bits[0]),
           "ROL": (2 * A - 256 * A7 + C, A7), "ROR": (lsrA + 128 * C, A_bits[0])}
    for mn, (res, cy) in acc.items():
        case([MN[mn], MODE["acc"], *S0], reg_tok("A", res, cy, V, 0, 1))
    rm_mem = 1 - r_io - (r_rom if mapper_board else 0)
    for mn, (res, cy) in rmw_res.items():
        tok = base("RM") + res + 256 * cy + 512 * a7 + 1024 * LEN
        case([MN[mn], direct, *S0, rm_mem], tok)
        case([MN[mn], is_["IN"], nNM], tok)
        if mapper_board:  # RMW on the board's registers: RM (not final), then MAP
            case([MN[mn], direct, *S0, r_rom], tok - 1024 * LEN)
    if cfg.board in (2, 3):  # the modified byte is written: it is the RM token's payload
        case([rmw, rm_mid, nNM], base("MAP") + rm_val + 2048 * LEN)
    if cfg.board == 1:  # MMC1 ignores the second (modified) write: the original byte counts
        mmc1_write([rmw, rm_mid, nNM], Mb[0], M7)
    case([rmw, direct, *S0, r_io], ior)
    # branches and jumps
    br = mns(*BRANCH_FLAG)
    case([br, *S0, taken, jt_rom], tokexpr("J", jt - 0x8000))
    case([br, *S0, taken, jt_ram], tokexpr("JL", jt))
    case([br, *S0, taken, 1 - jt_rom - jt_ram], S.tok("HALT"))
    case([br, *S0, 1 - taken], tokexpr("NT", 2))
    case([MN["JMP"], MODE["abs"], *S0, b2_7], tokexpr("J", absa - 0x8000))
    case([MN["JMP"], MODE["abs"], *S0, abs_ram], tokexpr("JL", absa))
    case([MN["JMP"], MODE["abs"], *S0, 1 - b2_7 - abs_ram], S.tok("HALT"))
    case([MODE["ind"], *S0, r_io], S.tok("HALT"))
    case([MODE["ind"], *S0, 1 - r_io], tokexpr("LO", M))
    case([MODE["ind"], is_["LO"], nNM, M7], tokexpr("J", ind_t - 0x8000))
    case([MODE["ind"], is_["LO"], nNM, ind_ram], tokexpr("JL", ind_t))
    case([MODE["ind"], is_["LO"], nNM, 1 - M7 - ind_ram], S.tok("HALT"))
    # JSR / RTS / RTI / BRK / NMI
    case([MN["JSR"], *S0, b2_7 + abs_ram], tokexpr("PSH", spm))
    case([MN["JSR"], *S0, 1 - b2_7 - abs_ram], S.tok("HALT"))
    case([MN["JSR"], is_["PSH"], JJ1, nNM], tokexpr("PSH", spm))
    case([MN["JSR"], is_["PSH"], JJ2, nNM, b2_7], tokexpr("J", absa - 0x8000))
    case([MN["JSR"], is_["PSH"], JJ2, nNM, abs_ram], tokexpr("JL", absa))
    case([mns("RTS", "RTI", "PLA", "PLP"), *S0], tokexpr("PUL", spp))
    case([MN["RTS"], is_["PUL"], J1, nNM], tokexpr("LO", M))
    case([MN["RTS"], is_["LO"], nNM], tokexpr("PUL", spp))
    case([MN["RTS"], is_["PUL"], J3, nNM, rts_rom], tokexpr("J", rts_t - 0x8000))
    case([MN["RTS"], is_["PUL"], J3, nNM, rts_ram], tokexpr("JL", rts_t))
    case([MN["RTS"], is_["PUL"], J3, nNM, 1 - rts_rom - rts_ram], S.tok("HALT"))
    case([MN["RTI"], is_["PUL"], J1, nNM], p_tok(Mb[0], Mb[1], Mb[2], Mb[3], M6, M7, 0, 0))
    case([MN["RTI"], is_["P"], 1 - f_fin, nNM], tokexpr("PUL", spp))  # (a final P: PLP etc.)
    case([MN["RTI"], is_["PUL"], J3, nNM], tokexpr("LO", M))
    case([MN["RTI"], is_["LO"], nNM], tokexpr("PUL", spp))
    case([MN["RTI"], is_["PUL"], J5, nNM, M7], tokexpr("J", ind_t - 0x8000))
    case([MN["RTI"], is_["PUL"], J5, nNM, ind_ram], tokexpr("JL", ind_t))
    case([MN["RTI"], is_["PUL"], J5, nNM, 1 - M7 - ind_ram], S.tok("HALT"))
    for intr, extra_l in ((MN["BRK"], [nNM]), (NM, [])):
        if intr is NM:
            case([is_["NMI"]], tokexpr("PSH", spm))
        else:
            case([intr, *S0], tokexpr("PSH", spm))
        case([intr, is_["PSH"], JJ1, *extra_l], tokexpr("PSH", spm))
        case([intr, is_["PSH"], JJ2, *extra_l], tokexpr("PSH", spm))
        case([intr, is_["PSH"], JJ3, *extra_l], tokexpr("LO", M))
        case([intr, is_["LO"], *extra_l, M7], tokexpr("JI", LO + 256 * M - 0x8000))
        case([intr, is_["LO"], *extra_l, 1 - M7], S.tok("HALT"))
    # stack, flags, transfers, inc/dec, NOP
    case([mns("PHA", "PHP"), *S0], tokexpr("PSH", spm + 256))
    case([MN["PLA"], is_["PUL"], nNM], reg_tok("A", M, C, V, 0, 1))
    case([MN["PLP"], is_["PUL"], nNM], p_tok(Mb[0], Mb[1], Mb[2], Mb[3], M6, M7, 0, 1))
    flags = {"C": C, "Z": Z, "I": I, "D": D, "V": V, "N": N}
    for mn, (fl, val) in FLAG_OPS.items():
        f2 = dict(flags, **{fl: val})
        case([MN[mn], *S0], p_tok(f2["C"], f2["Z"], f2["I"], f2["D"], f2["V"], f2["N"], 0, 1))
    for mn, dst, src in (("TAX", "X", A), ("TAY", "Y", A), ("TXA", "A", X), ("TYA", "A", Y),
                         ("TSX", "X", SP), ("INX", "X", incX), ("DEX", "X", decX),
                         ("INY", "Y", incY), ("DEY", "Y", decY)):
        case([MN[mn], *S0], reg_tok(dst, src, C, V, 0, 1))
    case([MN["TXS"], *S0], base("S") + X + 256)
    case([MN["NOP"], *S0], tokexpr("NT", 1))

    c.cases = cases  # (literals, token) per output case, for debugging
    c.output("next", c.sel(cases, "next"))
    return c
