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


def bench(name: str, frames: int, check: bool) -> dict | None:
    path = find(GAMES[name])
    if path is None:
        return None
    btn = parse_buttons(BUTTONS)
    t0 = time.perf_counter()
    toks, host = run_model(Rom.load(path), frames=frames, buttons=btn, keep_frames=False)
    dt = time.perf_counter() - t0
    row = {"game": name, "frames": host.frame, "tokens": len(toks), "seconds": round(dt, 2),
           "tok_s": round(len(toks) / dt), "fps": round(host.frame / dt, 2),
           "x_slower": round(60 * dt / host.frame, 1)}
    if check:
        ref, _, _ = run_reference(Rom.load(path), frames=frames, buttons=btn, keep_frames=False)
        n = min(len(ref), len(toks))  # one run may stop partway through the last record
        row["identical"] = ref[:n] == toks[:n] and abs(len(ref) - len(toks)) < 8
    return row


def show(rows: list[dict]) -> None:
    print(f"{'game':10} {'frames':>6} {'tokens':>10} {'sec':>7} {'tok/s':>7} {'fps':>6} "
          f"{'x NES':>6}  ok")
    for r in rows:
        print(f"{r['game']:10} {r['frames']:6} {r['tokens']:10,} {r['seconds']:7.1f} "
              f"{r['tok_s']:7,} {r['fps']:6.2f} {r['x_slower']:6.1f}  {r.get('identical', '-')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label", nargs="?")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--games", default=",".join(GAMES))
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = p.parse_args()
    if a.compare:
        before, after = (json.loads((OUT / f"{x}.json").read_text()) for x in a.compare)
        b = {r["game"]: r for r in before}
        print(f"{'game':10} {'fps before':>10} {'fps after':>10} {'speedup':>8}")
        for r in after:
            if r["game"] in b:
                print(f"{r['game']:10} {b[r['game']]['fps']:10.2f} {r['fps']:10.2f} "
                      f"{r['fps'] / b[r['game']]['fps']:7.2f}x")
        return
    rows = [r for g in a.games.split(",") if (r := bench(g, a.frames, not a.no_check))]
    show(rows)
    if a.label:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{a.label}.json").write_text(json.dumps(rows, indent=1))
        print(f"saved tmp/bench/{a.label}.json")


if __name__ == "__main__":
    main()
