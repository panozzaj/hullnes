"""Cartridge boards (mappers): how CPU addresses $8000-$FFFF reach PRG-ROM, and the CHR /
mirroring side the host needs.

A CPU store to $8000-$FFFF on a mapper board becomes a MAP token (see spec.py). Its payload
says what the write did, so the host and the circuit can both follow the board's state from
the trace alone:

  UxROM, CNROM   payload = the byte written (UxROM: PRG bank at $8000; CNROM: CHR bank)
  MMC1           payload = val5 | count << 5 | commit << 8 | reg << 9
                 a shift write: val5 = shift-register contents after it, count = bits so far;
                 a commit (5th write, or a reset with bit 7): register ``reg`` = val5

``write`` is the board's logic (used by the reference CPU), ``apply`` replays a token's payload
(used by everyone who reads the trace). The circuit (circuit.py) computes the same payloads.
"""

from dataclasses import dataclass, field

BANK = 0x4000


@dataclass
class Board:
    number: int
    prg: bytes  # all of PRG-ROM
    chr_size: int  # bytes of CHR (ROM or RAM)
    vertical: bool  # header mirroring (boards that don't control it)
    prgram: bool = False  # 8 KB of RAM at $6000-$7FFF

    @property
    def n16(self) -> int:
        return len(self.prg) // BANK

    def prg_index(self, a: int) -> int:
        raise NotImplementedError

    def write(self, ea: int, v: int) -> int:
        """A CPU store to $8000-$FFFF: update the board, return the MAP payload."""
        p = self.payload(ea, v)
        self.apply(p)
        return p

    def payload(self, ea: int, v: int) -> int:
        return v

    def apply(self, p: int) -> None:
        pass

    def chr_index(self, a: int) -> int:
        return a & 0x1FFF

    def nt_index(self, a: int) -> int:
        a &= 0x0FFF
        if self.vertical:
            return a & 0x07FF
        return (a & 0x03FF) | (a & 0x0800) >> 1

    def state_payloads(self) -> list[int]:
        """MAP payloads that put a fresh board into this board's (CPU-visible) state."""
        return []


@dataclass
class NROM(Board):
    def prg_index(self, a: int) -> int:
        return (a - 0x8000) % len(self.prg)


@dataclass
class UxROM(Board):
    bank: int = 0

    def prg_index(self, a: int) -> int:
        b = self.bank & (self.n16 - 1) if a < 0xC000 else self.n16 - 1
        return b * BANK + (a & 0x3FFF)

    def apply(self, p: int) -> None:
        self.bank = p & 255

    def state_payloads(self) -> list[int]:
        return [self.bank]


@dataclass
class CNROM(NROM):
    bank: int = 0

    def apply(self, p: int) -> None:
        self.bank = p & 255

    def chr_index(self, a: int) -> int:
        return (self.bank * 0x2000 + (a & 0x1FFF)) % self.chr_size

    def state_payloads(self) -> list[int]:
        return [self.bank]


def mmc1_payload(val5: int, count: int, commit: int, reg: int) -> int:
    return val5 | count << 5 | commit << 8 | reg << 9


@dataclass
class MMC1(Board):
    regs: list = field(default_factory=lambda: [0x0C, 0, 0, 0])  # control, chr0, chr1, prg
    shift: int = 0
    count: int = 0

    def payload(self, ea: int, v: int) -> int:
        if v & 0x80:
            return mmc1_payload(self.regs[0] | 0x0C, 0, 1, 0)
        val = self.shift | (v & 1) << self.count
        if self.count == 4:
            return mmc1_payload(val, 0, 1, ea >> 13 & 3)
        return mmc1_payload(val, self.count + 1, 0, 0)

    def apply(self, p: int) -> None:
        val5, count, commit, reg = p & 31, p >> 5 & 7, p >> 8 & 1, p >> 9 & 3
        if commit:
            self.regs[reg] = val5
            self.shift = self.count = 0
        else:
            self.shift, self.count = val5, count

    def prg_index(self, a: int) -> int:
        mode, p = self.regs[0] >> 2 & 3, self.regs[3] & 15 & (self.n16 - 1)
        if mode < 2:
            return (p & ~1) * BANK + (a & 0x7FFF)
        if mode == 2:
            b = 0 if a < 0xC000 else p
        else:
            b = p if a < 0xC000 else self.n16 - 1
        return b * BANK + (a & 0x3FFF)

    def chr_index(self, a: int) -> int:
        a &= 0x1FFF
        if self.regs[0] & 0x10:
            bank = self.regs[1] if a < 0x1000 else self.regs[2]
            return (bank * 0x1000 + (a & 0x0FFF)) % self.chr_size
        return ((self.regs[1] & ~1) * 0x1000 + a) % self.chr_size

    def nt_index(self, a: int) -> int:
        a &= 0x0FFF
        m = self.regs[0] & 3
        if m == 0:
            return a & 0x03FF
        if m == 1:
            return 0x400 | (a & 0x03FF)
        if m == 2:
            return a & 0x07FF
        return (a & 0x03FF) | (a & 0x0800) >> 1

    def state_payloads(self) -> list[int]:
        out = [mmc1_payload(self.regs[r], 0, 1, r) for r in range(4)]
        if self.count:
            out.append(mmc1_payload(self.shift, self.count, 0, 0))
        return out


BOARDS = {0: NROM, 1: MMC1, 2: UxROM, 3: CNROM}


def parse(data: bytes):
    """iNES file -> (board, chr bytes, chr_ram)."""
    if data[:4] != b"NES\x1a":
        raise ValueError("not an iNES file")
    nprg, nchr, f6, f7 = data[4], data[5], data[6], data[7]
    number = (f6 >> 4) | (f7 & 0xF0)
    if number not in BOARDS:
        raise ValueError(f"mapper {number} is not supported (0, 1, 2, 3)")
    off = 16 + (512 if f6 & 4 else 0)
    prg = data[off:off + BANK * nprg]
    if number == 0 and nprg == 1:
        prg = prg * 2
    chr_ = data[off + BANK * nprg:off + BANK * nprg + 8192 * nchr]
    if len(prg) & (len(prg) - 1) or len(prg) > 0x40000:
        raise ValueError(f"PRG size {len(prg)} not supported (a power of two up to 256 KB)")
    board = BOARDS[number](number, bytes(prg), len(chr_) or 8192, bool(f6 & 1),
                           prgram=number == 1)
    return board, bytearray(chr_ if nchr else bytes(8192)), nchr == 0
