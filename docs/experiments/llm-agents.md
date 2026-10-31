# LLM agents driving hullnes

Two runs on Super Mario Bros. through `hullnes session` (savestates as prompts, `step --why`
for which instruction last wrote an address). See [IDEAS.md](../IDEAS.md) for the motivation.

## Claude Code subagent: a RAM map

Task: build a RAM map of Super Mario Bros. purely from experiments, with evidence per entry and
a note wherever results disagree with what the model remembered. Result:
[smb-ram-map.md](smb-ram-map.md), about 45 address groups in about 35 tool calls.

What it reported about the tools:

- Branching one savestate with different inputs and diffing RAM found about half the map in one
  call (idle, walk, run and jump from the same state).
- `--why` explained causes rather than changes. The lives counter was a `DEC` at `$91D9` on the
  death frame; score, coin and timer digits share one writer (`$8F75 STA abs,Y`), which showed
  they are one display table; Mario's X and the enemies' X share `$BF40 STA zp,X`, which gave
  the layout of the object arrays.
- Its remembered score layout was off by one digit; its experiments corrected it.
- It found two bugs, both fixed: `step --png` returned a stale frame after a savestate round
  trip, and `--why` gave wrong answers in a step that crossed a context rollover.
- It asked for: every writer in a step (not only the last) with old values, a built-in RAM
  diff, a way to write RAM, and a per-frame watch on a few addresses.

## Gemma 4 (e4b, local via Ollama): one address

`scripts/llm_play.py`, task: find Mario's horizontal position. The first run failed 21 of 25
calls because the tool required frame ranges on buttons (`right@0-60`) and Gemma wrote
`right`; bare buttons are now accepted. The rerun found `$0086` in 11 calls (look, walk right,
read zero page, `why` on a range). Its stated evidence was partly wrong: it described a
decreasing value (the low byte wrapping as the level scrolled) as moving right.

Both models may know Super Mario Bros.'s RAM layout from training, so these runs show that the
tools work for an agent rather than measuring discovery from scratch. The Thwaite runs below are
the cleaner test.

## Thwaite, a game neither model knew

Thwaite is a homebrew Missile Command-style game. Both runs had no web access and no source.

**Claude Code subagent:** [thwaite-ram-map.md](thwaite-ram-map.md), about 45 regions (about 20
at high confidence) in about 50 tool calls, playing through the tutorial wave, the bonus tally
and a game over. Before starting it knew only the genre. It found the controller bytes, the
crosshair (`$04DE-$04ED`, whole-pixel X at `$04E0`), ammo per silo, one 20-slot object table
shared by the player's fireworks and the incoming missiles (both moved by the same code, which
`--why` showed), explosions, 12 building states including the two silos, and a score written
only at the end of a wave. It used `--why` to set aside every byte written by the sound engine.
It asked for read tracking, "run until this byte changes", a per-frame watch, and a built-in
diff.

**Gemma 4 (local):** asked for the crosshair's X. Its first run never looked at the screen
again after the start and spent 30 turns moving left and right on a tutorial dialog. With
`--auto-look` (the screen attached to every step) it got into the game and answered `$001A`
in 21 calls, without evidence. That is wrong. From one in-game savestate, holding Left for 20
frames moves `$04E0` from `$40` to `$1F` and holding Right moves it to `$60`, written by
`$8FE7 STA abs,X` on the last frame; `$001A` stays `$B6` either way, last written by the sound
code (`$C6CD INC (zp,X)`) 13 frames earlier. Gemma picked a byte that changed during its steps
without checking whether the input caused the change.
