# LLM + NES ideas

Things this design makes cheap that an ordinary emulator does not, ordered roughly from
"an afternoon on top of what exists" to "a research project". The common thread: the machine's
history is a token sequence, its memory is attention over that sequence, and its program is a
prompt. All three are things an LLM, or a tool an LLM drives, can read and edit.

## 1. Provenance: who wrote this byte? (exists)

`python -m hullnes why GAME.nes 0086 075A --frames 300` asks the RAM head which trace position
last wrote each address. That's one attention query, the same one the CPU does on every read,
with the writer's position as an extra value. In Super Mario Bros. it says that `$0086` (Mario's
X position) came from `STA $86,X` at `$BF40` in frame 299, with the add that produced it just
before.

Where it goes next: follow the chain. The writing instruction read its operands from somewhere,
and those reads are attention lookups too, so the whole dataflow graph behind a value is
recoverable. An agent that labels RAM ("$0086 is the player's X, updated every frame from
these two other bytes") could build a game's RAM map with no prior knowledge, which is the
first thing anyone hacking or speedrunning a game needs.

## 2. Glitch hunting from the trace

Several kinds of crash or exploit show up as specific tokens, so a hunter can scan the trace
instead of guessing from the screen:

| Signal | Token evidence | What it often means |
|---|---|---|
| Code running from RAM in a game that never does | `OPF` tokens | arbitrary code execution, a wrong-warp or memory-corruption setup |
| An unofficial opcode or a jump into I/O space | `HALT` | a crash: execution has left real code |
| Stack pointer wrapping | `PSH`/`PUL` payload going $00 -> $FF | stack overflow or underflow |
| Writes to unusual addresses | `ST` keys never seen in normal play | out-of-bounds table writes |
| A PC never seen in normal play | `J` targets | unreachable code (cut content, debug menus) |

The loop: record a baseline of normal play, fork the context at an interesting moment, try
many input sequences, flag any branch whose trace shows a signal the baseline lacks. Forking is
cheap because the PRG-ROM part of the prompt is already cached (`Session.prompt`). Branches
differ only in the RAM and register part, so a savestate is about 2,100 tokens (10,300 with
8 KB of cartridge RAM).

An LLM fits in two places: proposing input sequences from a description of the game ("try
clipping into the wall by ducking and jumping on the frame the screen scrolls"), and explaining
a flagged trace in words.

## 3. Savestates and patches as prompt edits

The prompt is the cartridge plus the machine state (`tokens.py`), so:

- A savestate is a prompt. Loading one is feeding it. You can also write one by hand: "Mario at
  X=200 in world 8-4 with 9 lives" is a few edited `RI` tokens, if you know the RAM map from
  idea 1.
- A Game Genie code is an edit to one `RB` token. An LLM can propose codes ("make jumps
  higher": find where vertical speed is set, try values), and the tool can check each proposal
  by running it and comparing traces.
- A ROM hack is a prompt edit as well, and the trace diff shows what it changed in execution,
  not only in the bytes.

## 4. An LLM that plays, with the console as its tool

The hybrid Percepta describes: a language model plans, and an exact executor runs. Here the
executor is a separate small model, so the language model sees frames, or the RAM map, at
decision points and emits buttons. Two cheap variants:

- Text only: describe the state from labelled RAM (from idea 1): "Mario at X=88, Goomba at
  X=140 moving left, 2 lives". Read the LLM's button choice for the next 30 frames.
- Search: let the LLM propose candidate plans, run each on a forked context, report results,
  and let it pick. That's tool-assisted speedrunning with an LLM as the strategist.

## 5. A learned emulator, trained on exact traces

The compiled model produces unlimited, exact training data: traces, prompts, and every hidden
activation. You can train a normal transformer (with 2D heads, or not) on the traces and see
which parts of the CPU it learns and where it fails: carries, rare opcodes, bank switching,
long-range RAM reads. Because the compiled model's layers have meaning, its activations can
supervise the learned model's layers, not only its outputs. This is the most direct way to
study Percepta's open question of how capable 2D-head models are when trained.

## 6. Put the PPU in the weights

The host still draws the picture. A PPU in the weights would need far more tokens per frame
(about 89k PPU dots against about 30k CPU cycles), but it is the same kind of circuit: pattern
fetches are ROM or CHR lookups, nametable and palette RAM are keyed memory, sprite evaluation
is a small search. A per-tile or per-scanline token format would keep the count manageable.
Then the model's output would be pixels, and "what is on screen" would be readable from the
trace by an LLM directly.

## Smaller things worth doing first

- MMC3 support (185 of the 684 ROMs in the local collection). It needs an IRQ from the host's
  scanline counter, injected like `NMI`.
- Speed: MMC1's 64 bank states each add gated units to every ROM query. Computing the two
  16 KB bank numbers once (one unit per state) and gating per bank would cut most of them.
- A dense export (`compile.py` from Qwenputer) to a `safetensors` checkpoint, and a check that a
  plain softmax forward pass reproduces short traces. That shows the weights are an ordinary
  transformer, independent of the Li Chao engine.
