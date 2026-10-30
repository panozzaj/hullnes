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
tools work for an agent rather than measuring discovery from scratch. A homebrew game the models
have not seen would be the cleaner test.
