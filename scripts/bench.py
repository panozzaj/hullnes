"""Benchmark: fixed games, inputs and frame counts; prints a table and saves it under a label.

    uv run scripts/bench.py LABEL [--frames 300] [--games smb,megaman,...]
    uv run scripts/bench.py --compare BEFORE AFTER

Commercial games come from NES_ROMS (a directory of .nes files) and are skipped when missing. Every run
also checks the trace token for token against the reference CPU (outside the timed part).
"""

import argparse
import json
import os
import time
from pathlib import Path

from hullnes.host import Rom
from hullnes.run import run_model, run_reference
from nesformer.host import parse_buttons

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "bench"
LIBRARY = Path(os.environ.get("NES_ROMS", "/nonexistent")).expanduser()
BUTTONS = "start@60-65+200-205+300-305,right@210-600,a@250-265+320-330+400-410"
GAMES = {
    "thwaite": ROOT / "roms" / "thwaite.nes",
    "smb": "Super Mario Bros. (World).nes",
    "megaman": "Mega Man (USA).nes",
    "zelda": "Legend of Zelda, The (USA) (Rev A).nes",
    "smb3": "Super Mario Bros. 3 (USA) (Rev A).nes",
}


def find(spec) -> Path | None:
    if isinstance(spec, Path):
        return spec
    hits = list(LIBRARY.rglob(spec)) if LIBRARY.exists() else []
    return hits[0] if hits else None


WARM = 30  # steady-state fps is timed from this frame on (after the prompt and JIT warm-up)


def bench(name: str, frames: int, check: bool, skip: bool = True) -> dict | None:
    path = find(GAMES[name])
    if path is None:
        return None
    btn = parse_buttons(BUTTONS)
    marks = {}

    def on_frame(host):
        if host.frame == WARM:
            marks["warm"] = time.perf_counter()

    t0 = time.perf_counter()
    toks, host = run_model(Rom.load(path), frames=frames, buttons=btn, keep_frames=False,
                           on_frame=on_frame, skip=skip)
    t1 = time.perf_counter()
    dt = t1 - t0
    steady = (host.frame - WARM) / (t1 - marks["warm"]) if "warm" in marks else 0.0
    row = {"game": name, "frames": host.frame, "tokens": len(toks), "seconds": round(dt, 2),
           "tok_s": round(len(toks) / dt), "fps": round(host.frame / dt, 2),
           "steady_fps": round(steady, 2), "x_slower": round(60 / steady, 2) if steady else 0,
           "skip": skip}
    if check:
        ref, _, _ = run_reference(Rom.load(path), frames=frames, buttons=btn, keep_frames=False,
                                  skip=skip)
        n = min(len(ref), len(toks))  # one run may stop partway through the last record
        row["identical"] = ref[:n] == toks[:n] and abs(len(ref) - len(toks)) < 8
    return row


def show(rows: list[dict]) -> None:
    print(f"{'game':10} {'frames':>6} {'tokens':>10} {'sec':>6} {'tok/s':>8} {'fps':>6} "
          f"{'steady':>7} {'x NES':>6}  ok")
    for r in rows:
        print(f"{r['game']:10} {r['frames']:6} {r['tokens']:10,} {r['seconds']:6.1f} "
              f"{r['tok_s']:8,} {r['fps']:6.2f} {r.get('steady_fps', 0):7.2f} "
              f"{r['x_slower']:6.2f}  {r.get('identical', '-')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label", nargs="?")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--games", default=",".join(GAMES))
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--no-skip", action="store_true", help="disable nesformer's idle-loop skip")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = p.parse_args()
    if a.compare:
        before, after = (json.loads((OUT / f"{x}.json").read_text()) for x in a.compare)
        b = {r["game"]: r for r in before}
        key = "steady_fps" if all("steady_fps" in r for r in before + after) else "fps"
        print(f"{'game':10} {'before':>10} {'after':>10} {'speedup':>8}   ({key})")
        for r in after:
            if r["game"] in b:
                print(f"{r['game']:10} {b[r['game']][key]:10.2f} {r[key]:10.2f} "
                      f"{r[key] / b[r['game']][key]:7.2f}x")
        return
    rows = [r for g in a.games.split(",") if (r := bench(g, a.frames, not a.no_check,
                                                         skip=not a.no_skip))]
    show(rows)
    if a.label:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{a.label}.json").write_text(json.dumps(rows, indent=1))
        print(f"saved tmp/bench/{a.label}.json")


if __name__ == "__main__":
    main()
