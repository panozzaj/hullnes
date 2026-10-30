"""nesformer's host (PPU, controllers, NMI timing), extended to mapper boards.

The host follows the board through MAP tokens in the trace, like everything else it knows
about the CPU. It needs the board for three things: the opcode at the PC (to count cycles),
CHR banks (the PPU's pattern tables), and nametable mirroring.
"""

from dataclasses import dataclass
from pathlib import Path

from hullnes import mapper
from nesformer import spec as S
from nesformer.host import Cart, Host

MAP_BASE = S.FAMILIES["MAP"][0]


class PrgView:
    """The CPU's view of $8000-$FFFF, indexed like nesformer's 32 KB prg (addr & 0x7FFF)."""

    def __init__(self, board):
        self.board = board

    def __getitem__(self, i: int) -> int:
        return self.board.prg[self.board.prg_index(0x8000 | (i & 0x7FFF))]

    def __len__(self) -> int:
        return 0x8000


class ChrView:
    """$0000-$1FFF of PPU space through the board's CHR banking."""

    def __init__(self, board, data: bytearray):
        self.board, self.data = board, data
        self.flat = board.number in (0, 2)  # no CHR banking: index directly

    def __getitem__(self, a: int) -> int:
        return self.data[a] if self.flat else self.data[self.board.chr_index(a)]

    def __setitem__(self, a: int, v: int) -> None:
        if self.flat:
            self.data[a] = v
        else:
            self.data[self.board.chr_index(a)] = v


@dataclass
class Rom:
    """A cartridge: the board (PRG banking, CHR, mirroring) and what the reference CPU needs."""

    data: bytes
    path: str = ""

    def __post_init__(self):
        self.board, _, _ = mapper.parse(self.data)

    @classmethod
    def load(cls, path) -> "Rom":
        return cls(Path(path).expanduser().read_bytes(), str(path))

    @property
    def number(self) -> int:
        return self.board.number

    @property
    def prg(self) -> bytes:
        return self.board.prg

    def fresh_board(self):
        return mapper.parse(self.data)[0]

    def host_cart(self) -> tuple[Cart, object]:
        board, chr_, chr_ram = mapper.parse(self.data)
        cart = Cart(PrgView(board), ChrView(board, chr_), board.vertical, chr_ram)
        return cart, board


class BoardHost(Host):
    """Host + a mapper board updated from MAP tokens."""

    board: object = None

    @classmethod
    def for_rom(cls, rom: Rom, **kw) -> "BoardHost":
        cart, board = rom.host_cart()
        h = cls(cart, **kw)
        h.board = board
        return h

    def feed(self, t: int) -> None:
        super().feed(t)  # the instruction's opcode was fetched before its write took effect
        if MAP_BASE <= t < MAP_BASE + S.FAMILIES["MAP"][1]:
            self.board.apply((t - MAP_BASE) & 0x7FF)

    def nt_index(self, a: int) -> int:
        return self.board.nt_index(a)
