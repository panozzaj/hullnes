# hullnes

An NES CPU compiled into the weights of a small vanilla transformer, decoded with an
O(log n) attention cache. It applies the construction from Percepta's
[Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers) to the NES
emulator in [nesformer](https://github.com/dbreunig/nesformer), and reuses most of
nesformer's trace format, reference CPU and host.

What runs in the weights: the 6502 core (all official opcodes), 2 KB of work RAM, 8 KB of
cartridge RAM, PRG-ROM of any size up to 256 KB, and the CPU side of four mapper boards
(NROM, UxROM, CNROM, MMC1). The PPU, controllers and NMI timing run in a host program, as in
nesformer. Nobody trained these weights; a compiler computed every one.

Checked token for token against a reference 6502: Super Mario Bros., Mega Man, Mega Man 2,
Metroid, The Legend of Zelda, Kid Icarus, Castlevania, Contra and DuckTales, plus nestest and
the two homebrew games nesformer ships.

## Results

Measured on one core of an Apple Silicon Mac. nesformer's numba engine runs at 3,617 tokens/s
on the same machine (its README reports about 1,650 on the machine it was built on).

| | nesformer | hullnes |
|---|---|---|
| Architecture | `Qwen3_5ForCausalLM` (7 Gated DeltaNet layers) | plain decoder, 10 layers, 2D hardmax heads, ReGLU |
| Memory | DeltaNet state rows, fixed size | the token trace, read by attention |
| Residual width | 768 | 455 (NROM) to 519 (MMC1, 256 KB) |
| Nonzero weights | 9.3M | 27k (NROM) to 48k (MMC1, 256 KB) |
| Attention heads | 7 layers of DeltaNet heads | 25 to 28 heads, 2 dims each |
| Tokens/s, engine alone | 3,617 | 25,000 (NROM), 24,000 (UxROM), 17,000 (MMC1) |
| Speed vs a real NES, with the host | 150-290x slower (its README) | 6x (Mega Man) to 11x (Super Mario Bros., Zelda) slower |
| Mappers | NROM | NROM, UxROM, CNROM, MMC1 |
| Code in RAM | halts | runs (Zelda and Kid Icarus need it) |

Checks (all in `tests/` or reproducible with `python -m hullnes check`):

- Teacher-forced on nestest and 60k tokens of each homebrew game, with a range check on the
  input of every piecewise-linear unit.
- Free-running 900k tokens of Full Game and Thwaite: every token and every rendered frame equal
  to the reference CPU's, including across forced context rollovers.
- Free-running 400k tokens of Super Mario Bros.: identical tokens and frames.
- Teacher-forced 300k to 600k tokens of each of the other commercial games above: 0
  mismatches.

## How it works

This section is written to be read in order. Each idea builds on the previous one.

### 1. The trace

The model emits one record per instruction, nesformer's format unchanged:
`LDX #$40` becomes `X($40 c0 v0 px0 L2)`, a store becomes `ST($00 a7=0 L2)`, a `JSR` becomes
`PSH PSH J($8279)`. Every token is a state change: a register's new value, a memory write, a
push, a jump. Most instructions are one token. The host injects two kinds of token: `IN` (the
answer to an I/O read) and `NMI` (vblank).

Tokens are never overwritten, so the current state of the machine is "the latest token that
wrote each thing". That makes every read a lookup into the past, which is what attention does.

### 2. Memory lookups with 2D hardmax heads

Each head here has a 2-dimensional query `(q, 1)` and key `(kx, ky)`, and returns the value of
the single position that maximizes `q*kx + ky` (hardmax; softmax at a high temperature gives
the same answer). Three lookups cover everything the CPU needs:

- **Latest of a family** (registers, flags): `kx = 0`, `ky = position - BIG*(1 - flag)`. The
  most recent flagged position wins. `A` is "the value of the latest `A` token".
- **Previous position**: `ky = -s^2`, `kx = 2s`, `q = t - 1`. The score
  `2s(t-1) - s^2 = (t-1)^2 - (s-(t-1))^2` peaks at `s = t - 1`.
- **Keyed memory**: a write at position `s` to address `k` stores the key line
  `(2Tk, -Tk^2 + s)`. A read of address `q` scores `T(q^2 - (k-q)^2) + s`. The exact address
  wins by a margin of at least `T`, and among writes to the same address the latest wins,
  because `T` exceeds every position. That is RAM.

The prompt holds the cartridge (`tokens.py`): PRG-ROM as 32 KB to 256 KB of byte tokens,
then the initial RAM, registers and PC. A ROM read is the previous-position trick with a
different target: `q = 1 + rom_offset` picks out the prompt token at that position.

### 3. The O(log n) attention cache

Each past position contributed a line `y = kx*q + ky`. A query asks which line is highest at
`q`. That is the upper envelope of a set of lines, the dual of a convex hull, and a
[Li Chao tree](https://cp-algorithms.com/geometry/convex_hull_trick.html) answers it in
O(log range) per insert and per query (`engine.py`). Percepta calls their version the
HullKVCache. Standard attention pays O(t) per token to score every cached key; here a token
costs the same at position 10 and position 1,000,000.

This matters for Percepta's WebAssembly interpreter, whose context grows without bound.
nesformer sidesteps the problem: Gated DeltaNet is recurrent, so its per-token cost is
already constant. The speedup here comes from the model being small (section 5), not from the
hull.

### 4. The program counter

Final tokens carry the instruction length. A cumulative-sum head (uniform attention times
the position, Percepta's construction) gives the bytes executed so far, and the PC is
`anchor + (bytes now - bytes at the anchor)`, where the anchor is the latest jump.

### 5. Arithmetic in ReGLU units

The feed-forward layers are ReGLU units `c * a * relu(b)`. On integer inputs:

- `relu(x + 1) - relu(x)` is the step function `[x >= 0]`.
- `relu(l1 + l2 - 1)` is AND of two 0/1 literals, so a case with five conditions is one unit.
- Any integer function `f(z)` on a range is the piecewise-linear interpolation of its values:
  one unit per change of slope. `z mod 256`, bit 7 of `z`, `z^2` and "is this address in I/O
  space" are all one FFN node.
- Adding `BIG*(gate - 1)` to every relu argument switches a whole function off unless the gate
  holds. Banked ROM queries use this: one gated function per bank state, summed.

The opcode decode costs nothing: a ROM byte's embedding already contains its meaning as an
opcode (mnemonic and addressing-mode one-hots), so the fetch head returns the decoded
instruction directly.

The output token is a sum over 135 to 160 mutually exclusive cases (depending on the board), each one unit:
`token_expression * relu(AND(conditions))`. The unembedding row for token `v` is
`2v*next - v^2`, so the logits peak at `v = next`.

### 6. Layer plan

| Layer | Attention | FFN |
|---|---|---|
| 1 | registers, flags, latches, mapper state, record start; byte cumsum | record stage, SP +- 1, (zp),Y address |
| 2 | PC anchor | PC, PC+1, PC+2 as banked ROM queries |
| 3 | fetch opcode and operands | addressing-mode candidates, branch test, return address |
| 4 | | read address, pushed byte, page-cross |
| 5 | | RAM and ROM queries, regions, write key |
| 6 | previous position: this token's write key | key squared, I/O register index |
| 7 | RAM read, ROM read | operand bits, adder, compares |
| 8 | | AND/ORA/EOR, overflow |
| 9 | | BIT's Z flag |
| 10 | | the next token |

A write's key has to be squared for its key line (`-Tk^2`), and a
square costs one unit per value, so a 10 KB key space would cost 10k units. The key travels
as two bytes instead: `k^2 = 65536h^2 + 512hl + l^2`, where `h*l` is a single ReGLU unit and
the squares are tables of at most 41 and 256 entries. A read of ROM and a read of RAM go to
different heads, and each head reads a zero cell when the address belongs to the other, so the
operand is `ram + rom` without a selection layer.

### 7. Mappers and code in RAM

A store to `$8000-$FFFF` on a mapper board becomes a `MAP` token whose payload says what the
write did. For MMC1 that includes the serial shift register (five writes per register), so
the model computes the shift state and the host reads CHR banks and mirroring from the same
tokens. Bank state is a few more "latest" heads, folded into the ROM queries.

Zelda and Kid Icarus run code from RAM. The opcode fetch happens in layer 3, before RAM write
keys exist (layer 6), so the model cannot attend to RAM code directly. Instead, a record whose
PC is in RAM starts with `OPF(opcode) OB1(byte) OB2(byte)`: the model reads its own code into
the trace through the ordinary data read, and small latch heads return those bytes (zeros in
any other record) to the same decode inputs the ROM fetch uses. ROM traces are unchanged.

### 8. Context rollover

Positions must stay below 2^20 so every score fits in a float64 exactly. Near the limit, the
driver reads the machine state out of the model (registers from their heads, RAM by querying
the RAM head's cache at every address) and starts a fresh context with that state as the
prompt. The PRG-ROM part of the prompt is cached and restored instead of re-fed. The trace
after a rollover matches the trace without one (tested with rollovers forced every 100k
tokens).

## Limitations

- Official opcodes only; an unofficial opcode emits `HALT`.
- No IRQs, so no MMC3 (the most common mapper in a typical collection) yet.
- MMC1 boards above 256 KB (SUROM: Dragon Warrior III and IV) are not supported.
- The PPU is in the host and renders per scanline, as in nesformer.
- No sound.

## Usage

```bash
uv sync
uv run python -m hullnes run roms/thwaite.nes --frames 600 \
  --buttons "start@90-95+200-205,a@230-233+250-253"
uv run python -m hullnes run ~/roms/smb.nes --frames 600 \
  --buttons "start@60-65,right@200-600,a@250-265+300-315"
uv run python -m hullnes check GAME.nes --tokens 300000   # free-run vs the reference CPU
uv run python -m hullnes trace GAME.nes --tokens 40
uv run python -m hullnes info GAME.nes                    # model size for that board
uv run pytest -q
```

`--buttons` follows nesformer's syntax: `button@first-last` frame ranges, `+` to combine.

## Files

| Path | Contents |
|---|---|
| `hullnes/circuit.py` | the CPU as a circuit graph (the model's "source code") |
| `hullnes/engine.py` | the forward pass with Li Chao tree attention caches (numba) |
| `hullnes/tokens.py` | vocabulary and prompt layout |
| `hullnes/mapper.py` | NROM, UxROM, CNROM, MMC1 |
| `hullnes/host.py` | nesformer's host plus mapper boards |
| `hullnes/run.py` | reference runs, teacher-forced check, free-running, rollover |
| `third_party/chipformer/dsl` | Qwenputer's circuit DSL (MIT), plus a `Cumsum` node |
| `third_party/nesformer` | nesformer's trace spec, reference CPU and host (MIT), extended for mappers and code in RAM |

## Credits

- [nesformer](https://github.com/dbreunig/nesformer) by Drew Breunig: trace format, reference
  CPU, host, test ROMs (MIT).
- [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal: the `chipformer` DSL,
  whose original target is this kind of 2D-head transformer (MIT).
- Percepta, [Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers): 2D
  heads, the hull cache, cumulative sums by uniform attention, the program as a prompt.
- Test ROMs: see `roms/SOURCES.md`.
