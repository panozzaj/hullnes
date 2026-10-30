"""Drivers: the reference CPU, the teacher-forced check, and free-running the transformer.

The host (PPU, controllers, NMI timing, the mapper board's CHR side) injects two kinds of token:
NMI at a record boundary when the PPU raised one, and IN right after the CPU emits IOR.
Everything else comes from the token source.
"""

import time
from functools import lru_cache

import numpy as np

from hullnes import tokens as T
from hullnes.circuit import Config, build
from hullnes.engine import Compiled, Engine
from hullnes.host import BoardHost, Rom
from nesformer import spec as S
from nesformer.cpu import Cpu

IOR_BASE = S.FAMILIES["IOR"][0]
ROLL_MARGIN = 4096  # roll the context over this many positions before it is full


@lru_cache(maxsize=8)
def compiled(cfg: Config = Config()) -> Compiled:
    return Compiled(build(cfg))


def new_engine(rom: Rom | None = None, check: bool = False,
               max_positions: int = 1 << 20) -> Engine:
    cfg = Config.for_board(rom.board) if rom is not None else Config()
    eng = Engine(compiled(cfg), max_positions=max_positions, check=check)
    eng.cfg = cfg
    return eng


def boot(rom: Rom, pc: int | None = None) -> Cpu:
    return Cpu.boot(rom.prg, pc, board=rom.fresh_board())


def run_reference(rom: Rom, frames: int = 60, buttons=None, max_tokens: int | None = None,
                  pc: int | None = None, skip: bool = True, keep_frames: bool = True):
    """Run nesformer's reference CPU against the host; returns (trace, host, cpu)."""
    host = BoardHost.for_rom(rom, buttons=buttons, keep_frames=keep_frames, skip=skip)
    cpu = boot(rom, pc)
    host.pc = cpu.pc
    toks = [S.tok("GO")]
    while not cpu.halted and host.frame < frames:
        if max_tokens is not None and len(toks) >= max_tokens:
            break
        if host.wants_nmi():
            toks.append(S.tok("NMI"))
            host.feed(toks[-1])
            gen = cpu.nmi_record()
        else:
            gen = cpu.record()
        send = None
        while True:
            try:
                t = gen.send(send)
            except StopIteration:
                break
            toks.append(t)
            host.feed(t)
            send = None
            if IOR_BASE <= t < IOR_BASE + 64:
                send = host.io_read(t - IOR_BASE)
                toks.append(S.tok("IN", send))
                host.feed(toks[-1])
    return toks, host, cpu


def injected(t: int) -> bool:
    return t == S.tok("NMI") or S.family(t)[0] == "IN"


class Session:
    """An engine bound to a cartridge, with prompt caching: the PRG part of the prompt is fed
    once and its attention cache is reused on every context rollover."""

    def __init__(self, rom: Rom, engine: Engine | None = None, check: bool = False):
        self.rom = rom
        self.eng = engine or new_engine(rom, check=check)
        self.lay = self.eng.cfg.layout
        self._rom_cache = None

    def prompt(self, state: dict) -> int:
        """Start a fresh context for ``state``; feeds the prompt up to GO and returns GO."""
        toks = self.lay.prompt(self.rom.prg, state)
        n = self.lay.ram_pos  # the zero entry + PRG
        if self._rom_cache is None:
            self.eng.reset()
            self.eng.feed(toks[:n])
            self._rom_cache = self.eng.save()
        else:
            self.eng.restore(self._rom_cache)
        self.eng.feed(toks[n:-1])
        return toks[-1]

    def boot_prompt(self, pc: int | None = None) -> int:
        return self.prompt(T.cpu_state(boot(self.rom, pc)))

    def machine_state(self) -> dict:
        """The CPU state as the model holds it after a record's final token: registers and flags
        from their heads, RAM / PRG-RAM by querying the RAM head's cache at every cell key."""
        e = self.eng
        d = e.dim
        n, z, c = (int(d(f"NZC[{i}]")) for i in range(3))
        flags = dict(N=n, Z=z, C=c, V=int(d("V[0]")), I=int(d("I[0]")), D=int(d("D[0]")))
        cells = e.peek("RAM", range(self.lay.n_ram))[:, 0].astype(np.int64)
        md = {k: int(e.v[next(iter(x.terms))]) for k, x in e.cp.g.map_dims.items()}
        board = e.cfg.board
        if board == 2:
            payloads = [md["bank"]]
        elif board == 1:
            from hullnes.mapper import mmc1_payload
            payloads = [mmc1_payload(md["ctrl"], 0, 1, 0), mmc1_payload(md["prg"], 0, 1, 3)]
            if md["count"]:
                payloads.append(mmc1_payload(md["shift"], md["count"], 0, 0))
        else:
            payloads = []
        return dict(A=int(d("A[0]")), X=int(d("X[0]")), Y=int(d("Y[0]")), SP=int(d("SP[0]")),
                    flags=flags, ram=bytes(cells[:0x800].tolist()),
                    prgram=bytes(cells[T.K_PRAM:].tolist()), map=payloads,
                    anchor=int(d("anchor[0]")), offset=int(d("bytes[0]") - d("anchor[1]")))

    def ram_key(self, addr: int) -> int:
        if addr < 0x2000:
            return addr & 0x7FF
        if self.lay.prgram and 0x6000 <= addr < 0x8000:
            return T.K_PRAM + addr - 0x6000
        raise ValueError(f"${addr:04X} is not RAM")

    def writer(self, addr: int) -> tuple[int, int]:
        """(value, context position of the token that wrote it) for a RAM address: one query of
        the RAM head's cache. A position inside the prompt means "set by the prompt"."""
        val, p = self.eng.peek("RAM", [self.ram_key(addr)])[0]
        return int(val), int(p)

    def rollover(self) -> int:
        """Compact the context: re-prompt with the current machine state. Call right after a
        final token. Returns the prediction for the next token."""
        return self.eng.step(self.prompt(self.machine_state()))


def check_trace(rom: Rom, trace: list[int], pc: int | None = None, check: bool = True,
                max_bad: int = 10):
    """Teacher-forced: feed the reference trace and compare the prediction at every position
    with the next token (skipping positions whose next token the host injects).
    Returns (positions checked, mismatches, tokens/s)."""
    ses = Session(rom, check=check)
    ses.boot_prompt(pc)
    t0 = time.perf_counter()
    pred = ses.eng.feed(trace)
    dt = time.perf_counter() - t0
    bad, checked = [], 0
    for i in range(len(trace) - 1):
        if injected(trace[i + 1]):
            continue
        checked += 1
        if pred[i] != trace[i + 1]:
            p = int(pred[i])
            bad.append((i, [T.name(x) for x in trace[max(0, i - 6):i + 1]], T.name(trace[i + 1]),
                        T.name(p) if 0 <= p < T.VOCAB else p))
            if len(bad) >= max_bad:
                break
    return checked, bad, len(trace) / dt if dt else 0.0


def run_model(rom: Rom, frames: int = 60, buttons=None, max_tokens: int | None = None,
              pc: int | None = None, expect: list[int] | None = None, skip: bool = True,
              keep_frames: bool = True, on_frame=None, engine: Engine | None = None,
              on_token=None):
    """Free-run the transformer: it predicts every token except the host's NMI / IN injections.
    With ``expect`` (a reference trace), stop at the first difference. Returns (trace, host)."""
    ses = Session(rom, engine)
    eng = ses.eng
    run_model.session = ses  # for tools that query the context after the run
    host = BoardHost.for_rom(rom, buttons=buttons, keep_frames=keep_frames, skip=skip)
    host.pc = boot(rom, pc).pc
    go = ses.boot_prompt(pc)
    toks = [go]
    pred = eng.step(go)
    halt, nmi = S.tok("HALT"), S.tok("NMI")
    seen = rollovers = 0
    while host.frame < frames and not host.halted:
        if max_tokens is not None and len(toks) >= max_tokens:
            break
        t = nmi if host.wants_nmi() else pred
        batch = [t]
        host.feed(t)
        if IOR_BASE <= t < IOR_BASE + 64:
            v = host.io_read(t - IOR_BASE)
            batch.append(S.tok("IN", v))
            host.feed(batch[-1])
        for tt in batch:
            toks.append(tt)
            if expect is not None and (len(toks) > len(expect) or expect[len(toks) - 1] != tt):
                want = expect[len(toks) - 1] if len(toks) <= len(expect) else None
                raise AssertionError(f"token {len(toks) - 1}: model {T.name(tt)}, reference "
                                     f"{T.name(want) if want is not None else 'end'}")
            pred = eng.step(tt)
            if on_token:
                on_token(tt, host, eng.pos - 1)
        if t == halt:
            break
        if eng.pos > eng.max_positions - ROLL_MARGIN and S.is_final(toks[-1]):
            pred = ses.rollover()
            rollovers += 1
        if on_frame and host.frame > seen:
            seen = host.frame
            on_frame(host)
    host.rollovers = rollovers
    return toks, host
