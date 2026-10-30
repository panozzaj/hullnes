"""Vocabulary and prompt layout.

The trace after GO is nesformer's format (third_party/nesformer/spec.py), plus MAP tokens for
writes to mapper boards. What differs from nesformer is how the machine starts. nesformer writes
the cartridge and registers into its DeltaNet state before decoding; here, as in Percepta's
construction, the program is part of the prompt:

    Z                       position 0: the null entry (what a ROM read of a non-ROM address sees:
                            byte 0, and no opcode, so RAM code decodes from its OPF token alone)
    RB(byte) x len(PRG)     all of PRG-ROM; byte i at position 1 + i
    RI(byte) x n_ram        the RAM head's cells, key = position - ram_pos:
                              0..2047 work RAM, 2048 the I/O port, 2049 zero, 2050.. PRG-RAM
    A(a) X(x) Y(y)          registers (LEN 0: not final, so the PC does not move)
    P(flags) S(sp)          flags, stack pointer
    MAP(p) ...              the mapper board's state (LEN 0)
    R / RL(anchor) NT(..)   the PC: anchor, then the bytes committed since it
    GO                      the trace starts

Every later token is generated (or injected by the host: NMI, IN).
"""

from dataclasses import dataclass

from nesformer import spec as S

RB_BASE = S.VOCAB
RI_BASE = S.VOCAB + 256
Z_TOK = S.VOCAB + 512
VOCAB = S.VOCAB + 513

K_IO, K_ZERO, K_PRAM = 0x800, 0x801, 0x802  # RAM-head keys


@dataclass(frozen=True)
class Layout:
    n_prg: int = 0x8000
    prgram: bool = False

    rom_pos = 1

    @property
    def ram_pos(self) -> int:
        return 1 + self.n_prg

    @property
    def n_ram(self) -> int:
        return K_PRAM + (0x2000 if self.prgram else 0)

    def prompt(self, prg: bytes, st: dict) -> list[int]:
        """st: A X Y SP flags ram prgram map (payloads) anchor offset."""
        assert len(prg) == self.n_prg and len(st["ram"]) == 0x800
        f = st["flags"]
        toks = [Z_TOK] + [RB_BASE + b for b in prg]
        cells = bytes(st["ram"]) + b"\0\0" + (bytes(st["prgram"]) if self.prgram else b"")
        toks += [RI_BASE + b for b in cells]
        toks += [S.t_reg(r, st[r], f["C"], f["V"], 0, 0) for r in ("A", "X", "Y")]
        toks += [S.t_p(f, 0, 0), S.tok("S", st["SP"])]
        toks += [S.tok("MAP", p) for p in st.get("map", [])]
        a = st["anchor"]
        toks.append(S.tok("R", a - 0x8000) if a >= 0x8000 else S.tok("RL", a))
        off = st["offset"]
        toks += [S.tok("NT", 3)] * (off // 3) + ([S.tok("NT", off % 3)] if off % 3 else [])
        toks.append(S.tok("GO"))
        return toks


def family(t: int) -> tuple[str, int]:
    if t == Z_TOK:
        return "Z", 0
    if t >= RI_BASE:
        return "RI", t - RI_BASE
    if t >= RB_BASE:
        return "RB", t - RB_BASE
    return S.family(t)


def name(t: int) -> str:
    fam, p = family(t)
    if fam in ("RB", "RI"):
        return f"{fam}(${p:02X})"
    if fam == "Z":
        return "Z"
    return S.name(t)


def cpu_state(cpu) -> dict:
    """The prompt state of a nesformer reference Cpu (with its board, if any)."""
    return dict(A=cpu.A, X=cpu.X, Y=cpu.Y, SP=cpu.SP, flags=dict(cpu.flags), ram=bytes(cpu.ram),
                prgram=bytes(cpu.prgram) if cpu.prgram is not None else b"",
                map=cpu.board.state_payloads() if cpu.board is not None else [],
                anchor=cpu.anchor, offset=cpu.offset)
