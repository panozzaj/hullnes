<!-- Produced by a Claude Code subagent (Opus 5.5) driving `hullnes session` on Thwaite, a homebrew
     game, without the web or the game's source; about 50 tool calls. Branch and file names refer
     to its working directory, which is not in the repo. -->

# Thwaite RAM map, built from experiments in hullnes

Built by branching savestates with `hullnes session new/step/ram`, diffing RAM dumps, asking `--why` about
the bytes that changed, and checking screenshots. I did not read the ROM bytes or the game's source code.

## What I knew beforehand

- Thwaite is a Missile Command-style homebrew game by Damian Yerrick: you move a crosshair and fire
  fireworks from two silos to protect a row of houses. I had a vague memory that levels are "hours"
  across a week of "days".
- I knew nothing about its RAM layout, code addresses or data structures.
- Prior knowledge had little effect. The in-game tutorial says "Move the crosshair with the Control Pad,
  and shoot a firework with B or A", which gave the controls anyway. The only entries it shaped are the
  day/hour guesses for `$03DD-$03DF` (marked **[prior]**).

## How the game behaved (observed)

- Boot with `start@90-95+200-205`: at frame 300 a tutorial dialog is up. A few A presses lead into
  hour 1 (`SUN 01:00`, a tutorial wave of slow missiles).
- HUD: left silo ammo, right silo ammo, score, then `DAY HH MM`.
- B fires from the left silo and A from the right one.
- Points are awarded only at the end of a wave: "Nice Job! Left: 2 [houses] 26 [missiles] Bonus: 4600"
  (consistent with 1000 per house plus 100 per missile). Shooting an enemy gave no immediate points.
- Silos can be destroyed. Game over came when the second silo fell, with one house still standing.

## RAM map

Writer PCs come from `--why` (the instruction that last wrote the byte during a step).

| Address | Meaning | Evidence | Writer PC (`--why`) | Confidence |
|---|---|---|---|---|
| `$0000` | NMI/vblank handshake flag (guess) | Rewritten every frame | `$C2F7` STA zp | Low |
| `$0010-$002F` | Sound engine: channel data pointers (`$12/13`, `$16/17`, `$1A/1B`, `$20/21`...) and per-channel duration counters (`$1C`, `$2F`) | Pointer pairs point at `$C8xx-$CBxx` and are advanced with INC; a hit set `$14/15`=`$C889` (a sound effect) | `$C549`, `$C568` DEC, `$C58D`, `$C6CD` INC | Medium |
| `$0030` | Frame counter | Rose by `$14` in 20 frames, `$2C` per 300 frames (mod 256) | `$8001` INC zp (the NMI handler starts at `$8000`?) | High |
| `$0031` | Some UI/state sub-mode (04 during play, 00/30/24 during the tally) | Diffs only | `$8EEE` STA, `$80C9` STX | Low |
| `$0034` | Controller 1, current buttons. Right=`01`, Left=`02`, Down=`04`, Up=`08` (standard order, so A=`80`, B=`40`, Select=`20`, Start=`10`) | One-direction branches gave exactly these bits; left+up = `0A` | `$C367` STA zp,X (X=player) | High |
| `$0035` | Controller 2, current buttons | Same writer as `$0034` with X=1 | `$C367` | Medium |
| `$0036-$0037` | Per-player derived button bytes (probably newly pressed) | Written right after `$34/35` each frame | `$C36F` STA zp,X | Low |
| `$0038` | Game state: `02` playing a wave, `03` end-of-wave tally, `07` game over | Changed exactly at wave end, the next wave and game over | `$8261` STA zp (at wave end) | Medium-high |
| `$003F-$0042` | Random number generator state (32-bit?) | Changes in seemingly random ways during play and stays put during the tally (no spawns) | Not captured | Medium-low |
| `$0043-$004D` | A shuffled permutation of building indices `0..$0B` (11 entries), probably the target order | Contents always a permutation, reshuffled over time | Not captured | Low |
| `$004E` | Unknown wave parameter (05 in wave 1, 08 at wave end, 05/04 in wave 2) | Diffs only | Not captured | Low |
| `$0100-$01BF` | VRAM update / text buffer in the stack page (PPU address+length records, then dialog text in ASCII: "firework with B or A.") | Readable in the RAM dump at boot | Not checked | Medium |
| `$0200-$02FF` | Sprite table (OAM). Sprite 0 is the crosshair: Y = `$04E4`-5, X = `$04E0`-4 | Moved with the D-pad | DMA source | High |
| `$0300-$030B` | Small buffer written by `$C0xx` code (maybe a PPU/sound queue) | `--why` only | `$C064` STA, `$C0E9` STX | Low |
| `$030C-$033B` | Smoke-trail particle pool, X (48 entries) | Same index `$25` written by the same spawn routine in all four rows; values track missile trails | `$84B8` | Medium |
| `$033C-$036B` | Particle Y | Same | `$84BD` | Medium |
| `$036C-$039B` | Particle type/frame (02/04 seen) | Same | `$84AE` | Medium-low |
| `$039C-$03CB` | Particle lifetime/countdown (`A0` = idle) | Decreasing by 4; updated every frame | `$84FF` STA abs,X | Medium |
| `$03CC-$03D7` | Building state, 12 entries left to right (10 houses; index 2 = left silo, index 9 = right silo). `01` intact, `02` hit (collapsing), `00` destroyed | House at x≈`$96` went `01`→`02`→`00` as the PNG showed it destroyed; silo entries hit 00 as each silo's ammo went to 0 | Not captured | High |
| `$03D8` | Buildings destroyed this wave (silos included); reset each wave | Counted 1..8 in wave 1, reset to 0, 3 at game over | `$8C80` INC abs | High |
| `$03DA` | Score / 100 (4600 shown, `$2E` = 46). The high-order byte is probably `$03D9` (00) | Written once, the frame after the state went to 03 | `$88C4` STA abs | Medium |
| `$03DB` | Unknown, constant `02` (difficulty? player count?) | Never changed | Not written | Low |
| `$03DD` | Day of week (`00` = SUN) **[prior]** | Constant 00, HUD SUN | Not written | Low |
| `$03DE`, `$03DF` | Hour/wave counters. Both went 00→01 when the HUD went `01 00`→`02 01`; one is the hour (shown +1) **[prior]** | One transition only, so I couldn't separate them | Not captured | Low-medium |
| `$03E0` | Seconds elapsed in the current wave | +1 per ~60 frames, reset to ~04 at the next wave | `$8907` INC abs | Medium-high |
| `$03E1`, `$03E2` | Sub-second dividers feeding `$03E0` (cycle and wrap) | Cycle through small values, INC every frame/few frames | `$88E6`, `$88D0` INC abs | Medium |
| `$03E3` | Unknown, 00 | Written periodically | `$863D` STA | Low |
| `$03E4` | Wave parameter (05 tutorial, 06 hour 2; set to 02 at the tally, 01 at game over) | Written at wave end | `$87E8` STA abs | Low |
| `$03E5` | Countdown timer (falls during play and the tally) | DEC each step | `$85C1` DEC abs | Low-medium |
| `$03E6` | Left silo ammo (B button) | `0F`→`0E` on B; HUD 15; 00 once the silo is destroyed | `$9054` DEC abs,X | High |
| `$03E7` | Right silo ammo (A button) | `0C`→`0B` on A; HUD 12; 00 once destroyed | `$9054` DEC abs,X | High |
| **Object table** | 20 slots, one row per field, row stride `$14`. Slots 0-1 hold `FF`/`07` (unknown, maybe per-player), slots 2-5 are player fireworks, slots 6+ are enemy missiles | The same motion code updates a firework (slot 2) and an enemy (slot 6) | Motion: `$8C0B-$8C27`; launch: `$9054-$910B` | High (layout) |
| `$03E8+i` | Object X (whole pixels) | Firework went BC→67 in 24 frames toward the target; enemy matched its PNG position | `$8C14` | High |
| `$03FC+i` | X fraction | Low byte of 8.8 X; moves by the dX fraction each frame | `$8C0B` | High |
| `$0410+i` | Object Y. Y=0 seems to mean a dead/free slot | Firework BE→94; a killed enemy got Y=00 | `$8C27` | High |
| `$0424+i` | Y fraction | | `$8C1E` | High |
| `$0438+i`, `$044C+i` | dX, 8.8 signed (int, frac) | Firework `FC.77` ≈ -3.5 px/frame matches the X motion | `$90F2`, `$910B` (at launch) | High |
| `$0460+i`, `$0474+i` | dY, 8.8 signed | Firework `FE.3F`; enemy `00.30` | `$90C6`, `$90CB` | High |
| `$0488+i` | Unknown per object (firework: 12 from the right silo, 18 from the left; enemy: 05-0C) | Set at launch | `$90A3` | Low |
| `$049C+i` | Unknown, always 00 | | `$9063` STA abs,Y (launch) | Low |
| `$04B0+i` | Firework: target X (= crosshair X at fire time). Enemy: target building index 0-11 | Firework=40 with cursor X 40; enemy values 08/01/00/0A, and the X drift matched building 8 | `$9073` | High (firework), Medium (enemy) |
| `$04C4+i` | Firework: frames left until it bursts (decrements by 1/frame). Enemy: smoke-puff emission timer | 23,20,1D... per 3 frames | `$8C42` DEC; enemy `$8CBF` DEC | High (firework) |
| `$04D8-$04DD` | Firework target Y (slots 0-5 only; the row is cut short by the cursor block) | =80 with cursor Y 80; =40 later | `$9078` | High |
| `$04DE-$04ED` | Crosshair, 2-entry arrays (player 1, player 2): `$04DE/F` X fraction, `$04E0/1` X, `$04E2/3` Y fraction, `$04E4/5` Y, `$04E6/7` dX frac, `$04E8/9` dX int, `$04EA/B` dY frac, `$04EC/D` dY int. The cursor accelerates | Directional branches moved exactly these; OAM sprite 0 follows | `$8FB2`, `$8FE7`, `$8FF1`, `$9026`, `$8F86`, `$8F8B`, `$8FA3`, `$8FA8` STA abs,X | High |
| `$04EE` | Unknown (26/2A/31 seen) | Diffs only | Not captured | Low |
| `$04F0-$04F7` | Explosion X (8 slots) | = firework target X when it burst; a second explosion appeared at the killed enemy's X | Not captured | High |
| `$04F8-$04FF` | Explosion Y | Same; `C8` = ground impact | Not captured | High |
| `$0500-$0507` | Explosion timer/radius phase (`FF` = free) | `1A` for a firework burst, `16` for an enemy's death burst, counting down | Not captured | Medium-high |
| `$0508-$0513` | Retargeting table: for each of the 12 building columns, the X of the nearest surviving building | Starts as building X positions 10 20 38 50 ... E0 and converges as buildings die | `$942D` STA abs,X | Medium |
| `$0514-$051F` | Same table: the index of the nearest surviving building | Starts 00..0B, collapses toward survivors (e.g. three entries become 08) | `$9467` STA abs,X | Medium |
| `$0520-$0533` | 20-entry per-object array (unknown) | `$0533` changed when buildings were hit | `$930F` STA abs,Y | Low |
| `$0535-$057F` | Sound engine state: per-channel arrays at stride 4 (`$0540/44/48/4C` countdowns), a 16-bit tempo/position accumulator at `$056E/F`, `$057C` | Writers are all in the `$C5xx-$C7xx` sound code | `$C502`, `$C53B`, `$C5F5`, `$C5FE`, `$C621`, `$C6FB`, `$C708`, `$C71D` | Medium |
| `$0580-$07FF` | Unused (all zero throughout) | | | High |

## How the tools helped, and what was missing

**What worked well**

- **Savestate branching plus RAM diff is the core loop.** Five 20-frame branches from one state (nothing,
  left, right, up, down) isolated the controller byte and the crosshair block in a single diff. A and B
  branches found ammo and the firework slot in one more. Running branches in parallel with `&` made this cheap.
- **`--why` was the most valuable tool.** One query gave each byte's writer PC and frame, which:
  - showed that the firework and the enemy are moved by the same code (`$8C0B-$8C27`), turning "two
    lists" into "one 20-slot object table";
  - separated game logic (`$8xxx-$9xxx`) from the sound engine (`$C5xx-$C7xx`) and I/O (`$C0xx-$C3xx`)
    by PC range, which let me discard a lot of noise quickly;
  - pinned timing: the score byte was written once, the frame after the state byte went to `03`
    (`$8261` then `$88C4`), and `$03D8` was incremented by `$8C80` the frame before.
- **PNGs** were needed to tie numbers to meaning: the house x position, HUD values, "Nice Job!" and
  bonus text, and "GAME OVER".

**What was missing or awkward**

- **No read tracing.** `--why` says who wrote a byte, not who reads it. Bytes that only change at
  scripted moments (`$03DB`, `$03DD`, building states) gave no PC at all. A "who read this" query, or
  `--why` for the step where the write happened (when it's earlier), would help.
- **"Not written during this step (or written before a context rollover)" is ambiguous.** In a long
  step I couldn't tell "constant" from "written but forgotten".
- **No watch/break-on-change.** To find the frame where a house died I had to bisect with several
  300-frame steps. A `--until ADDR!=value` or "stop when RAM byte changes" option would turn several
  steps into one.
- **No per-frame RAM trace for a few addresses.** I chained eight 3-frame steps to get a time series
  of one object slot. A `--trace 03ED,0411 --every 1` that prints values per frame would help a lot.
- **Speed.** About 3 s for 20 frames, and 30-60 s per 300 frames later in the run. The tutorial wave
  needed about 3600 frames idle to finish. Long idle runs had to go in the background.
- **No disassembly at a PC.** Writer PCs are very useful, but seeing a few instructions around them
  (e.g. `$88C4`: is the bonus added in binary or BCD?) would settle questions like the score encoding.
  That would mean reading ROM bytes, which this exercise ruled out, but a real user would want it.
- `ram` output is hex text. I wrote a small diff/grid script (`diff.py`, `grid.py`). A built-in
  `session diff a.pkl b.pkl c.pkl` that excludes OAM and the stack would save setup on every game.

**Discovery without prior knowledge vs a known game**

It felt different mainly in how hypotheses formed. With a known game you look up the expected
variables and confirm them. Here every structure had to emerge from diffs, and the layout was not what
I would have guessed: one shared 20-slot struct-of-arrays table for fireworks and enemies, 2-entry
per-player crosshair arrays sitting right after it, buildings that include the silos, and a score
that changes only at wave end. That last one cost a few branches: I shot down an enemy, saw no score
change, and only found the score byte after a whole wave finished. The game's own tutorial text and
HUD did most of what prior knowledge would have done. The `--why` PC ranges were what made exploring
a game blind workable: once you know `$C5xx` is sound, you can ignore a third of every diff.

## Files

All in `tmp/agent_thwaite/`: savestates `*.pkl`, screenshots `*.png`, RAM dumps `*.txt`, and helpers
`diff.py` (multi-dump diff, skips stack and OAM), `grid.py` (prints an array table), `br.sh` (branch + dump).
