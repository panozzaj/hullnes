"""uv run pytest -q"""

import os
from pathlib import Path

import pytest

from hullnes.run import check_trace, new_engine, run_model, run_reference
from nesformer import spec as S
from nesformer.cpu import OPS, Cpu
from hullnes.host import Rom
from nesformer.host import parse_buttons

ROMS = Path(__file__).resolve().parent.parent / "roms"
BUTTONS = "start@90-95+200-205,right@100-400,a@130-135+160-165"


def cart(name):
    return Rom.load(ROMS / name)


def test_reference_matches_nestest_log():
    """nesformer's reference CPU (the trace we must reproduce) matches nestest's golden log."""
    prg = cart("nestest.nes").prg
    cpu = Cpu.boot(prg, 0xC000)
    n = 0
    for line in (ROMS / "nestest.log").read_text().splitlines():
        pc = int(line[:4], 16)
        if pc < 0x8000 or int(line[6:8], 16) not in OPS:
            break
        while cpu.offset >= S.OFF_MAX:
            list(cpu.record())
        want = [int(line[line.index(k + ":") + len(k) + 1:][:2], 16)
                for k in ("A", "X", "Y", "P", "SP")]
        assert (cpu.pc, [cpu.A, cpu.X, cpu.Y, cpu.p_byte(0), cpu.SP]) == (pc, want), line
        g = cpu.record()
        t = next(g)
        try:
            while True:
                t = g.send(0xFF if S.family(t)[0] == "IOR" else None)
        except StopIteration:
            pass
        n += 1
    assert n == 3348


@pytest.mark.parametrize("name,pc,n", [("nestest.nes", 0xC000, 5000),
                                        ("full_game.nes", None, 60000),
                                        ("thwaite.nes", None, 60000)])
def test_teacher_forced(name, pc, n):
    """The circuit predicts every reference token (with range checks on every pwl input)."""
    c = cart(name)
    trace, _, _ = run_reference(c, frames=10**6, pc=pc, max_tokens=n,
                                buttons=parse_buttons(BUTTONS))
    checked, bad, _ = check_trace(c, trace, pc=pc)
    assert checked > 0 and not bad, bad[:3]


def test_detects_a_wrong_token():
    c = cart("nestest.nes")
    trace, _, _ = run_reference(c, frames=10**6, pc=0xC000, max_tokens=2000)
    i = next(k for k, t in enumerate(trace) if k > 500 and S.family(t)[0] == "A")
    trace[i] ^= 1
    _, bad, _ = check_trace(c, trace, pc=0xC000)
    assert bad and bad[0][0] == i - 1


@pytest.mark.parametrize("name", ["full_game.nes", "thwaite.nes"])
def test_free_run_with_rollover(name):
    """Free-running with a small context (forcing rollovers) reproduces the trace and frames."""
    btn = parse_buttons(BUTTONS)
    ref, rhost, _ = run_reference(cart(name), frames=10**6, max_tokens=250_000, buttons=btn)
    toks, host = run_model(cart(name), frames=10**6, max_tokens=250_000, buttons=btn,
                           expect=ref, engine=new_engine(cart(name), max_positions=100_000))
    assert host.rollovers >= 2
    assert [bytes(r) for f in host.frames for r in f] == [bytes(r) for f in rhost.frames
                                                          for r in f]


# Commercial cartridges are not in the repo; these run when NES_ROMS (a directory of .nes files) has them.
LIBRARY = Path(os.environ.get("NES_ROMS", "/nonexistent")).expanduser()
GAMES = ["Super Mario Bros. (World).nes", "Mega Man (USA).nes", "Metroid (USA).nes",
         "Mega Man 2 (USA).nes", "Legend of Zelda, The (USA) (Rev A).nes", "Mega Man 3 (USA).nes",
         "Kirby's Adventure (USA) (Rev A).nes"]


def find_game(name):
    hits = list(LIBRARY.rglob(name)) if LIBRARY.exists() else []
    if not hits:
        pytest.skip(f"{name} not in {LIBRARY}")
    return Rom.load(hits[0])


@pytest.mark.parametrize("name", GAMES)
def test_library_teacher_forced(name):
    rom = find_game(name)
    trace, _, cpu = run_reference(rom, frames=10**6, max_tokens=150_000,
                                  buttons=parse_buttons("start@120-125+200-205+300-305"))
    assert not cpu.halted
    checked, bad, _ = check_trace(rom, trace)
    assert checked > 0 and not bad, bad[:3]


@pytest.mark.skipif(not os.environ.get("HULLNES_SLOW"), reason="slow (set HULLNES_SLOW=1)")
def test_dense_softmax_model():
    """The weights as an ordinary softmax transformer reproduce the trace (about 2 minutes)."""
    from hullnes.dense import verify

    checked, bad = verify(cart("nestest.nes"), 1500, pc=0xC000, log=lambda *a: None)
    assert checked > 1000 and bad == 0
