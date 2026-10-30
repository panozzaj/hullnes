"""Step-by-step play with savestate files, for agents (an LLM with a shell) and for people.

    hullnes session new  ROM --frames 120 --buttons "start@60-65" --out s0.pkl [--png s0.png]
    hullnes session step s0.pkl --frames 60 --buttons "right@0-60" --out s1.pkl [--png s1.png]
                         [--why 0086,075A]
    hullnes session ram  s1.pkl 0080-009F 075A

A savestate holds the CPU state as the model holds it (registers, flags, RAM, PRG-RAM, board;
read out of the attention caches, as a context rollover does) and a copy of the host (PPU,
timing, controller). ``step`` starts a fresh context whose prompt is that state, so the machine
continues exactly. Button frame ranges in ``step`` count from the start of the step.
"""

import copy
import pickle
from pathlib import Path

from hullnes import tokens as T
from hullnes.host import Rom
from hullnes.run import Session, run_model
from nesformer import spec as S
from nesformer.cpu import OPS
from nesformer.host import frame_rgb, parse_buttons


def normalize_buttons(spec: str, frames: int) -> str:
    """Accept 'right' or 'right+b' (held for the whole step) as well as 'right@0-60'."""
    parts = []
    for part in filter(None, (x.strip() for x in (spec or "").split(","))):
        parts.append(part if "@" in part else f"{part}@0-{frames}")
    return ",".join(parts)


def parse_addrs(spec: str, limit: int = 32) -> list[int]:
    """'0086, 075A, 0090-0093' -> addresses (at most ``limit``)."""
    out = []
    for part in filter(None, (x.strip().replace("$", "") for x in str(spec or "").split(","))):
        lo, _, hi = part.partition("-")
        lo = int(lo, 16)
        out.extend(range(lo, (int(hi, 16) if hi else lo) + 1))
    return out[:limit]


def _run(rom_path: str, state, host, frames: int, buttons: str, why=()):
    """Run ``frames`` frames (from the savestate, or from power-on), stop at the first record
    boundary after, and return (new savestate, writers of the ``why`` addresses)."""
    rom = Rom.load(rom_path)
    ses = Session(rom)
    first = host.frame if host is not None else 0
    target = first + frames
    buttons = normalize_buttons(buttons, frames)
    script = parse_buttons(buttons) if buttons else None
    btn = (lambda f: script(f - first)) if script else None
    out = {}
    log = []  # (frame, record pc, token) per context position
    cur = {"pc": None}

    def on_token(t, h, p):
        if cur["pc"] is None:
            cur["pc"] = h.pc
        if p < len(log):  # a context rollover restarted the positions
            del log[p:]
            log[:] = [None] * p  # what came before is now the prompt
        while len(log) < p:
            log.append(None)
        log.append((h.frame, cur["pc"], t))
        if S.is_final(t):
            cur["pc"] = h.pc
            if h.frame >= target:
                out["state"] = ses.machine_state()
                out["host"] = copy.deepcopy(h)
                return True
        return False

    start = (state, copy.deepcopy(host)) if state is not None else None
    run_model(rom, frames=target + 1, buttons=btn, on_token=on_token, session=ses, start=start,
              keep_frames=True, fast=False)
    h = out["host"]
    frames_kept = h.frames[-1:]  # keep only the last frame in the savestate
    h.frames = frames_kept
    h.buttons = None  # each step sets its own
    writers = {}
    for a in why:
        val, p = ses.writer(a)
        if p < len(log) and log[p] is not None:
            f, pc, t = log[p]
            op = h.cart.prg[pc & 0x7FFF] if pc >= 0x8000 else None
            mn = f"{OPS[op][0]} ({OPS[op][1]})" if op in OPS else "?"
            writers[a] = f"${val:02X}, written at frame {f} by the instruction at ${pc:04X} {mn}"
        else:
            writers[a] = (f"${val:02X}, not written during this step (or written before a "
                          f"context rollover inside it)")
    return {"rom": rom_path, "state": out["state"], "host": h}, writers


def save(sv, path, png=None):
    Path(path).write_bytes(pickle.dumps(sv))
    if png and sv["host"].frames:
        from PIL import Image

        Image.frombytes("RGB", (256, 240), frame_rgb(sv["host"].frames[-1])).resize(
            (512, 480), Image.NEAREST).save(png)


def load(path):
    return pickle.loads(Path(path).read_bytes())


def describe(sv) -> str:
    st, h = sv["state"], sv["host"]
    f = st["flags"]
    pc = st["anchor"] + st["offset"]
    flags = "".join(k if f[k] else k.lower() for k in "NVDIZC")
    return (f"frame {h.frame}  PC ${pc & 0xFFFF:04X}  A ${st['A']:02X} X ${st['X']:02X} "
            f"Y ${st['Y']:02X} SP ${st['SP']:02X} {flags}")


def ram_dump(sv, spec: str) -> str:
    st = sv["state"]
    lines = []
    for part in spec:
        lo, _, hi = part.partition("-")
        lo = int(lo.replace("$", ""), 16)
        hi = int(hi.replace("$", ""), 16) if hi else lo
        for row in range(lo & ~15, hi + 1, 16):
            cells = []
            for a in range(row, row + 16):
                if lo <= a <= hi:
                    if a < 0x2000:
                        cells.append(f"{st['ram'][a & 0x7FF]:02X}")
                    elif 0x6000 <= a < 0x8000 and st["prgram"]:
                        cells.append(f"{st['prgram'][a - 0x6000]:02X}")
                    else:
                        cells.append("..")
                else:
                    cells.append("  ")
            lines.append(f"${row:04X}: " + " ".join(cells))
    return "\n".join(lines)


def add_parser(sub):
    p = sub.add_parser("session", help="step-by-step play with savestate files (for agents)")
    ss = p.add_subparsers(dest="scmd", required=True)
    n = ss.add_parser("new")
    n.add_argument("rom")
    n.add_argument("--frames", type=int, default=60)
    n.add_argument("--buttons", default="")
    n.add_argument("--out", required=True)
    n.add_argument("--png")
    s = ss.add_parser("step")
    s.add_argument("state")
    s.add_argument("--frames", type=int, default=30)
    s.add_argument("--buttons", default="", help="frame ranges count from the step's start")
    s.add_argument("--out", required=True)
    s.add_argument("--png")
    s.add_argument("--why", default="", help="comma-separated RAM addresses (hex)")
    r = ss.add_parser("ram")
    r.add_argument("state")
    r.add_argument("ranges", nargs="+", help="hex address or range, e.g. 0080-009F 075A")
    p.set_defaults(fn=cmd)


def cmd(a):
    if a.scmd == "new":
        sv, _ = _run(a.rom, None, None, a.frames, a.buttons)
        save(sv, a.out, a.png)
        print(describe(sv))
    elif a.scmd == "step":
        old = load(a.state)
        why = parse_addrs(a.why)
        sv, writers = _run(old["rom"], old["state"], old["host"], a.frames, a.buttons, why)
        save(sv, a.out, a.png)
        print(describe(sv))
        for addr, text in writers.items():
            print(f"  ${addr:04X} = {text}")
    else:
        sv = load(a.state)
        print(describe(sv))
        print(ram_dump(sv, a.ranges))
