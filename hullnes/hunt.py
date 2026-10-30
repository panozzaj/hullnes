"""Glitch hunting: fork the machine at a frame and try many inputs, watching the trace.

A savestate here is a prompt: the machine state read out of the model (registers, flags, RAM,
board) becomes the prompt of a fresh context, so a fork costs one prompt (about 2,100 tokens,
plus 8 KB of cartridge RAM where the board has it) on top of the cached PRG-ROM. The host
(PPU, timing) is copied as it was.

What gets flagged is visible in the trace itself, compared with a baseline of normal play:

  HALT        an unofficial opcode or a jump into I/O space (a crash)
  OPF         code running from RAM (arbitrary code execution, if the game never does it)
  stack       the stack pointer leaving the range seen in normal play
  new code    jump targets never seen in normal play (usually just new gameplay; sometimes
              unreachable code)
"""

import copy
import random
from dataclasses import dataclass, field

from hullnes.run import Session, run_model
from nesformer import spec as S
from nesformer.host import BUTTONS


@dataclass
class Watch:
    targets: set = field(default_factory=set)
    ram_pcs: set = field(default_factory=set)
    halted: bool = False
    sp_lo: int = 255
    sp_hi: int = 0
    tokens: int = 0

    def __call__(self, t, host, p):
        self.tokens += 1
        fam, pay = S.family(t)
        if fam in ("J", "JL", "JI", "JIL"):
            self.targets.add(S.anchor_addr(fam, pay))
        elif fam == "OPF":
            self.ram_pcs.add(host.pc)
        elif fam == "HALT":
            self.halted = True
        if fam in ("PSH", "PUL", "S"):
            sp = pay & 255
            self.sp_lo, self.sp_hi = min(self.sp_lo, sp), max(self.sp_hi, sp)


def random_buttons(seed: int, start: int, frames: int, hold=(4, 20),
                   names=("a", "b", "up", "down", "left", "right")):
    """A random controller script: a new random combination every few frames."""
    rng = random.Random(seed)
    plan, f = [], start
    while f < start + frames:
        n = rng.randint(*hold)
        mask = 0
        for nm in names:
            if rng.random() < 0.3:
                mask |= BUTTONS[nm]
        if mask & BUTTONS["left"] and mask & BUTTONS["right"] and rng.random() < 0.5:
            mask &= ~BUTTONS["left"]
        plan.append((f, f + n, mask))
        f += n

    def fn(frame):
        for a, b, m in plan:
            if a <= frame < b:
                return m
        return 0
    return fn


def hunt(rom, fork_frame: int, frames: int, branches: int, buttons=None, seed: int = 0,
         log=print):
    """Play to ``fork_frame`` (with ``buttons``), then run ``branches`` random input scripts for
    ``frames`` frames each. Returns (baseline Watch, [(seed, Watch, host)])."""
    base = Watch()
    fork = {}
    ses = Session(rom)

    def on_token(t, host, p):
        base(t, host, p)
        if not fork and host.frame >= fork_frame and S.is_final(t):
            fork["state"] = ses.machine_state()
            fork["host"] = copy.deepcopy(host)
            fork["host"].frames = []

    run_model(rom, frames=fork_frame + frames, buttons=buttons, on_token=on_token, session=ses,
              keep_frames=False)
    log(f"baseline: {base.tokens:,} tokens to frame {fork_frame + frames}, "
        f"{len(base.targets)} jump targets, stack ${base.sp_lo:02X}-${base.sp_hi:02X}, "
        f"code in RAM at {len(base.ram_pcs)} PCs")
    results = []
    for b in range(branches):
        s = seed + b
        w = Watch()
        host = copy.deepcopy(fork["host"])
        host.keep_frames = True
        run_model(rom, frames=fork_frame + frames,
                  buttons=random_buttons(s, fork_frame, frames), on_token=w, session=ses,
                  start=(fork["state"], host))
        results.append((s, w, host))
        log(f"  branch {b} (seed {s}): " + describe(w, base))
    return base, results


def flags(w: Watch, base: Watch) -> dict:
    return {
        "halt": w.halted,
        "ram_code": len(w.ram_pcs - base.ram_pcs),
        "stack": w.sp_lo < base.sp_lo or w.sp_hi > base.sp_hi,
        "new_code": len(w.targets - base.targets),
    }


def describe(w: Watch, base: Watch) -> str:
    f = flags(w, base)
    out = [f"{w.tokens:,} tokens"]
    if f["halt"]:
        out.append("HALT")
    if f["ram_code"]:
        out.append(f"code in RAM at {f['ram_code']} new PCs")
    if f["stack"]:
        out.append(f"stack ${w.sp_lo:02X}-${w.sp_hi:02X} outside normal play")
    out.append(f"{f['new_code']} new jump targets")
    return ", ".join(out)


def score(w: Watch, base: Watch) -> tuple:
    f = flags(w, base)
    return (f["halt"], f["ram_code"], f["stack"], f["new_code"])
