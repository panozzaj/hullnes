"""hullnes command line.

    uv run python -m hullnes run GAME.nes [--frames 600] [--buttons SCRIPT] [--gif out.gif]
    uv run python -m hullnes check GAME.nes [--tokens 300000]     free-run vs the reference CPU
    uv run python -m hullnes trace GAME.nes [--tokens 40]
    uv run python -m hullnes info
"""

import argparse
import time
from pathlib import Path

from hullnes import tokens as T
from hullnes.circuit import Config
from hullnes.host import Rom
from hullnes.run import compiled, run_model, run_reference
from nesformer.host import frame_rgb, parse_buttons


def load_cart(path: str) -> Rom:
    try:
        return Rom.load(path)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")


def save_gif(frames, path: str, every: int, scale: int) -> None:
    from PIL import Image

    ims = [Image.frombytes("RGB", (256, 240), frame_rgb(f)) for f in frames[::every]]
    if scale > 1:
        ims = [im.resize((256 * scale, 240 * scale), Image.NEAREST) for im in ims]
    if ims:
        ims[0].save(path, save_all=True, append_images=ims[1:], loop=0,
                    duration=int(1000 * every / 60))


def cmd_run(a):
    cart = load_cart(a.rom)
    t0 = time.perf_counter()
    last = [t0]

    def on_frame(host):
        if host.frame % 60 == 0:
            now = time.perf_counter()
            print(f"  frame {host.frame}: {60 / (now - last[0]):.1f} fps", flush=True)
            last[0] = now

    toks, host = run_model(cart, frames=a.frames, buttons=parse_buttons(a.buttons or ""),
                           max_tokens=a.max_tokens, skip=not a.no_skip, on_frame=on_frame)
    dt = time.perf_counter() - t0
    print(f"{len(toks):,} tokens, {host.frame} frames in {dt:.1f}s: {len(toks) / dt:,.0f} tok/s, "
          f"{host.frame / dt:.1f} fps ({60 * dt / max(host.frame, 1):.1f}x slower than an NES), "
          f"{host.rollovers} context rollovers")
    out = a.gif or Path(a.rom).with_suffix(".gif").name
    save_gif(host.frames, out, a.every, a.scale)
    print(f"wrote {out}")


def cmd_check(a):
    cart = load_cart(a.rom)
    btn = parse_buttons(a.buttons or "")
    ref, rhost, _ = run_reference(load_cart(a.rom), frames=10**9, buttons=btn,
                                  max_tokens=a.tokens, pc=a.pc)
    toks, host = run_model(cart, frames=10**9, buttons=btn, max_tokens=a.tokens, expect=ref,
                           pc=a.pc)
    same = all(bytes(x) == bytes(y) for fa, fb in zip(host.frames, rhost.frames)
               for x, y in zip(fa, fb))
    print(f"{len(toks):,} tokens identical to the reference CPU; {host.frame} frames, "
          f"frames identical: {same}")


def cmd_trace(a):
    cart = load_cart(a.rom)
    toks, _ = run_model(cart, frames=10**9, max_tokens=a.tokens, pc=a.pc)
    print(" ".join(T.name(t) for t in toks))


def cmd_why(a):
    """Run, then ask the RAM head who last wrote each address (one attention query each)."""
    from nesformer import spec as S
    from nesformer.cpu import OPS

    rom = load_cart(a.rom)
    log_frame, log_pc, log_tok = [], [], []
    cur = {"pc": boot_pc(rom)}

    def on_token(t, host, p):
        if p < len(log_tok):  # a rollover restarted the context
            del log_frame[p:], log_pc[p:], log_tok[p:]
        while len(log_tok) < p:
            log_frame.append(-1), log_pc.append(-1), log_tok.append(-1)
        log_frame.append(host.frame), log_pc.append(cur["pc"]), log_tok.append(t)
        if S.is_final(t):
            cur["pc"] = host.pc  # the next record's PC

    _, host = run_model(rom, frames=a.frames, buttons=parse_buttons(a.buttons or ""),
                        on_token=on_token)
    ses = run_model.session
    for addr in a.addr:
        val, p = ses.writer(addr)
        if p >= len(log_tok) or log_tok[p] < 0:
            print(f"${addr:04X} = ${val:02X}: set by the prompt (initial RAM, or before the latest "
                  f"context rollover)")
            continue
        pc = log_pc[p]
        op = host.cart.prg[pc & 0x7FFF] if pc >= 0x8000 else None
        mn = f"{OPS[op][0]} ({OPS[op][1]})" if op in OPS else "?"
        ctx = " ".join(T.name(x) for x in log_tok[max(0, p - 3):p + 1] if x >= 0)
        print(f"${addr:04X} = ${val:02X}: written at frame {log_frame[p]} by the instruction at "
              f"${pc:04X} ({mn}); trace ... {ctx}")


def boot_pc(rom) -> int:
    from hullnes.run import boot
    return boot(rom).pc


def cmd_info(a):
    cp = compiled(Config.for_board(load_cart(a.rom).board) if a.rom else Config())
    print(cp.stats())
    print("heads:", ", ".join(cp.head_names))


def main(argv=None):
    p = argparse.ArgumentParser(prog="hullnes")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("rom")
    r.add_argument("--frames", type=int, default=120)
    r.add_argument("--buttons", default="")
    r.add_argument("--gif")
    r.add_argument("--every", type=int, default=2)
    r.add_argument("--scale", type=int, default=2)
    r.add_argument("--max-tokens", type=int)
    r.add_argument("--no-skip", action="store_true", help="run every iteration of wait loops")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("check")
    c.add_argument("rom")
    c.add_argument("--tokens", type=int, default=300_000)
    c.add_argument("--buttons", default="")
    c.add_argument("--pc", type=lambda s: int(s, 16))
    c.set_defaults(fn=cmd_check)
    t = sub.add_parser("trace")
    t.add_argument("rom")
    t.add_argument("--tokens", type=int, default=40)
    t.add_argument("--pc", type=lambda s: int(s, 16))
    t.set_defaults(fn=cmd_trace)
    y = sub.add_parser("why", help="who last wrote these RAM addresses?")
    y.add_argument("rom")
    y.add_argument("addr", nargs="+", type=lambda s: int(s.replace("$", ""), 16))
    y.add_argument("--frames", type=int, default=120)
    y.add_argument("--buttons", default="")
    y.set_defaults(fn=cmd_why)
    i = sub.add_parser("info")
    i.add_argument("rom", nargs="?")
    i.set_defaults(fn=cmd_info)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
