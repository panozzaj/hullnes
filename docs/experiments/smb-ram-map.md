<!-- Produced by a Claude Code subagent (Opus 5.5) driving `hullnes session` on Super Mario Bros.;
     about 35 tool calls. Branch and file names refer to its working directory, which is not in
     the repo (savestates contain game RAM). The two tool bugs it reports at the end are fixed. -->

# Super Mario Bros. RAM map, built with hullnes session tools

All states start from `s0.pkl` (boot + `start@40-45`, 300 frames; World 1-1, Mario at X=$28, standing). Branch names below are the savestate files in this directory (`*.pkl`, RAM dumps in `*.ram`, step logs with `--why` output in `*.log`). "Writer" is the `--why` answer (PC and mnemonic of the last write in that step). Values are hex.

Key branches:

| branch | from | input | what happened |
|---|---|---|---|
| idle / right / run / jump | s0 | 60 frames: nothing / right / right+B / A@0-15 | baseline movement diffs |
| jmid | s0 | A held 12 frames | Mario mid-jump |
| walk150 | s0 | right 150 frames | Mario at $EA, screen scrolled, first Goomba spawned |
| touch | walk150 | right 60 frames | walked into the Goomba, died |
| dead | touch | 260 idle frames | lives decremented, level restarting |
| st22 | walk150 | right, A@22-26 | stomped the Goomba (+100) |
| blockw | walk150 | right@0-8, A@6-30 | hit first ? block: coin +1, +200 |
| mb30 -> mr | st22 -> land | left, jump under brick-row ? block, then walk right | mushroom spawned, then collected (+1000, Mario big) |
| farA / farB / farpic | mr | right+B with jumps | three Goombas on screen near X=$300 |

Screenshots confirming meaning: `s0.png`, `w450.png` (=walk150), `blockpic.png` (=blockw), `mrpic.png` (=mr), `farpic.png` (=farB). These were rendered by replaying the full input script with `session new`, because `step --png` returns a stale frame (see bottom).

## Table

| Address | Meaning | Evidence | Writer (--why) | Confidence |
|---|---|---|---|---|
| $0009 | Frame counter | 0 at s0, $3C after 60 frames, $96 after 150 | $8119 INC zp (every frame) | high |
| $06FC | Joypad 1 this frame (A=$80, B=$40, Right=$01) | right:01, run:41, jmid:80, idle:00 | $8E7B STA abx | high |
| $074A | Joypad copy (same values as $06FC) | identical to $06FC in every branch | not queried | medium |
| $000A / $000D | A/B button state bits ($80=A, $40=B); $000D likely previous frame | run: 40/40, jmid: 80/80 | not queried | medium |
| $0086 | Mario X, low byte (world coordinates) | 28 idle, 63 walk, 84 run after 60 frames; EA after 150 frames walking | $BF40 STA zpx (shared object-move routine, X=0 for Mario) | high |
| $006D | Mario X, page (high byte) | 0 -> 1 when $0086 wraps past FF (st22: $144; farpic: $316) | $BF46 STA zpx | high |
| $03AD | Mario X relative to screen left edge | = Mario X - (071A:071C) in all 9 states checked; pinned at $70 while scrolling right | not queried | high |
| $00CE | Mario Y (top of sprite; $B0 = standing on ground, smaller = higher) | jmid: 8A; touch (death hop): A4; back to B0 after landing | $BFED STA zpx (in air); $DD4B STA zp (ground snap) | high |
| $00B5 | Mario vertical page (1 = on screen) | 1 in all play states | not queried | low |
| $009F | Mario vertical speed (signed; negative = up) | jmid: FD; st22 stomp bounce: FF; touch death hop: FC; 0 on ground | $C002 STA zpx; $DD52 on landing | high |
| $0433 | Vertical speed fractional part | jmid 60, 0 on ground | $BFFB STA abx | medium |
| $0416 | Y sub-pixel | jmid E0, 0 on ground | $BFDF STA abx | medium |
| $001D | Mario airborne state (0 ground, 1 jumping/airborne, 2 seen when falling) | jmid 1, ground 0, farA (in air) 2 | $B4B9 STA zp at jump start; $DD5C on landing; $D94A STY in death | high for 0/1, low for 2 |
| $0706, $0707, $0708, $0709, $070A | Jump setup: set on the frame A is pressed; $0708 = Y at takeoff (B0), $0709/$070A = gravity values (20/70 on standing jump, 60/60 on running jump) | all change only in jump branches, all written frame 301 | $B4D4, $B4E7, $B4ED STA abs | medium |
| $0057 | Mario horizontal speed (signed) | walk 18, run 28, 0 standing, turning in air 28 -> 24 | $B5F7 / $B60B STA zp | high |
| $0700 | Absolute horizontal speed | walk 18, run 28 (same as $0057 moving right) | $B620 STA abs | medium |
| $0705, $0400 | Horizontal sub-pixel / speed fraction | change every frame while moving, 0 standing | not queried | low |
| $0033 | Mario facing direction (1 right, 2 left) | 2 after walking left (mb30), 1 otherwise; did not change when pressing left in mid-air | not written in mid-air test | medium |
| $0045 | Mario moving direction (0 none at boot, 1 right, 2 left) | 0 at s0, 1 after right, 2 in mb30 | $B14A STA zp | medium |
| $000E | Player state: $08 normal, $09 growing (mushroom), $0B dying, $00 during level reload | touch: 0B (writer at death frame 491); mr: 09; dead: 00 | $D948 STA zp (death), $B27A STA zp (back to 08) | high |
| $0747 | Gameplay freeze timer (counts down while game is frozen for death / growth) | touch: EC; mr: E8; 0 normal | $90DC STA (ind),Y in reset | medium |
| $0754 | Player size (1 = small, 0 = big) | 1 until mushroom, 0 after (mr) | $B265 STA abs | high |
| $0756 | Powerup level (0 small, 1 super) | 0 -> 1 on mushroom pickup | $D842 STA abs (pickup frame 757) | high |
| $075A | Lives minus one (start = 2, shown as 3) | 2 -> 1 after death | $91D9 DEC abs, frame 700 | high |
| $075E | Coin count | 0 -> 1 when ? block hit | $BC0C INC abs | high |
| $07ED / $07EE | Coin display digits (tens / ones) | 00 -> 01 on coin; HUD shows x01 | $8F75 STA aby (shared digit writer) | high |
| $07DE-$07E2 | Score display digits; $07E0 = thousands, $07E1 = hundreds, $07E2 = tens; the HUD's last 0 is not stored | stomp -> 07E1=1 (HUD 000100 not captured but +100 known); coin block -> 07E1=2 with HUD "000200"; mushroom -> 07E0=1, 07E1=1 with HUD "001100" | $8F75 STA aby | high for 07E0/07E1, medium for the rest |
| $07DD | Probably the top score digit (above what the HUD shows) | stayed 0 | none | low. **Disagrees with my prior** that 07DD-07E2 is a 6-digit score ending in ones: what I saw shows 07E1 is hundreds |
| $07D7-$07DC | Top score digits (copy of score: $07DA tracks $07E0, $07DB tracks $07E1) | blockw: 07DB=2; mr: 07DA=1 | $8FB2 STA aby, one frame after the score write | medium |
| $07F8 / $07F9 / $07FA | Timer digits (hundreds / tens / ones) | s0 3,9,7 = HUD 397; walk150 3,9,1 = HUD 391; blockpic 3,8,9 = HUD 389; mrpic 3,7,8 = HUD 378. Ticks once per ~24 frames | $8F75 STA aby | high |
| $0787 | Timer tick countdown (reloads at $18) | cycles C/18/5/11...; written in sync with timer digits | not queried | low |
| $07A0 | Countdown before level starts (3 at s0 -> 0; 4 on lives screen after death) | dead: 04 | $8113 DEC abx (timer-decrement loop) | medium |
| $071C / $071A | Screen left edge X (low byte / page) | Mario X - left edge = $03AD in all 9 states; camera follows Mario only to the right | $AFD4 STA abs | high |
| $071D / $071B | Screen right edge X (low / page) = left edge + $FF | 071D = 071C - 1 and 071B = 071A + 1 in all states (FF/00 at start) | $B03E STA abs | high |
| $073F | Horizontal scroll value (= $071C) | equal to $071C in every state | $AFD7 STA abs | high |
| $0778 | PPU control mirror (nametable bit toggles as screen pages advance) | 10 -> 11 -> 15 -> 14 | not queried | low |
| $0725, $0726 | Level-render column position (page, column) | increase as the screen scrolls | not queried | low |
| $0770 | Operation mode (1 = in game) | 1 after death reset | $927E STA abs | low (only one value seen) |
| $0772 | Operation task within mode (3 in play, 1 while reloading after death) | 3 -> 1 | $905D INC abs | medium |
| $075F / $0760 | World / level (0-based) | both 0 in World 1-1; never changed | none | low, unverified (I could not reach another level) |
| $000F-$0013 | Enemy slot active flags (slots 0-4) | 000F=1 when first Goomba appears; 000F-0011 = 1 with three Goombas (farpic) | not written in the steps queried | high |
| $0016-$001A | Enemy type per slot ($06 = Goomba) | 0016=06 when first Goomba spawns; 0016-0018 = 06 with three Goombas on screen | $C208 STA zpx (spawn) | high |
| $006E-$0072 | Enemy X page per slot | Goomba at $145 (006E=1); farpic slots 2/3/3 | $C1AE STA zpx (spawn), $BF46 (move) | high |
| $0087-$008B | Enemy X low byte per slot | slot0 45 -> 27 as it walks left; farpic slots at $2C4, $318, $330 match the three Goombas in the screenshot | $BF40 STA zpx (same routine as Mario X, X register = slot+1); $C1B4 STA zpx on spawn | high |
| $00CF-$00D3 | Enemy Y per slot ($B8 = Goomba on ground) | 00CF=B8 at spawn, 00CF-00D1 = B8 with three Goombas | $C277 STA zpx | high |
| $0058-$005C | Enemy X speed (signed) | 0058 = F8 (-8, walking left) | $CAC5 STA zpx | high |
| $0046 | Enemy 0 direction (2 = left) | 2 while Goomba walks left | not queried | medium |
| $001E | Enemy 0 state (0 walking, 4 stomped/flattened) | st22: 04 after stomp, others 0 | $D9D6 STA zpx at stomp frame | high |
| $03AE | Enemy 0 X relative to screen | CC = $145 - $7A | not queried | medium |
| $001B / $0014 / $008C / $0073 / $0039 | Powerup object slot: type ($2E), active flag, X low, X page, powerup kind (0 = mushroom) | set when brick-row ? block hit (mb30); cleared on pickup (mr) | $BC4B / $BC64 / $BC79 STA zp (spawn); $C99A-$C99E (despawn) | high for slot/type/active/X; medium for $0039 |
| $0032, $009B, $0076, $0082, $00D7, $00E3 | Bumped-block / spinning-coin objects after a ? block hit | only change in blockw | $BB76 STA aby ($0032), $BB44 STA aby ($009B) | low |
| $04AC-$04AF | Mario hitbox (x1, y1, x2, y2 on screen) | 04AC = 03AD+3, 04AE = 03AD+D; Y follows $00CE | not queried | medium |
| $04B0-$04B3 | Enemy 0 hitbox | follows Goomba screen X | not queried | medium |
| $06D5 | Mario sprite graphics frame (B8 standing, 60/68 walk, 80 jump, B0 dying) | changes with animation | not queried | medium |
| $07A7-$07AD | Pseudo-random number bytes | change every frame regardless of input | not queried | medium |
| $00F0-$00FF | Sound engine variables | change with jumps, stomps, death music | $F586 STY zp ($00F2 on coin) | low |
| $0200-$02FF | Sprite OAM buffer (Y, tile, attr, X); Mario's X shows up in 0207/020B... | track $03AD | not queried | high |
| $0300-$0360 | PPU/VRAM update buffer | changes as columns stream in | not queried | medium |

## How the tools helped / what was missing

**Most useful calls**

- The first 4-way branch from one savestate (idle / right / right+B / jump for 60 frames), diffed together, gave about half the table in one call: X position, speed, scroll, joypad, frame counter, timer, and the jump variables. Comparing four columns side by side shows quickly which bytes depend on which input.
- Event branches from `walk150` (stomp / touch / ? block / mushroom) gave the game-state bytes: score digits, coins, lives, player state, powerup, enemy state. One clean event per branch kept the diffs short.
- The `new`-from-power-on screenshots were needed to tie digits to the HUD. That is how I found that `$07E1` is the hundreds digit, not tens as I remembered.
- Savestates are deterministic: replaying the full input script from boot gave byte-identical RAM to the chained savestates (checked twice: 450 and 780 frames).

**What `--why` added over RAM diffs**

- **It separates cause from coincidence.** A diff says `$075A` went 2 -> 1; `--why` says `DEC $075A` at `$91D9` on frame 700, which is a deliberate decrement and not a side effect.
- **It groups addresses by routine.** `$07E1`, `$07EE` and `$07FA` all come from `$8F75 STA aby`, so score, coin and timer digits share one digit writer, a sign that they belong to one display-digit table. `$0086` (Mario X) and `$0087` (enemy X) share `$BF40 STA zpx`, so the objects are stored as parallel arrays indexed by X register, which is how I worked out the enemy slot layout. The top-score copy (`$8FB2`, one frame later) is clearly separate from the score (`$8F75`).
- **The frame number tells you when.** `$001D`, `$0706-$070A` written at frame 301 (the frame A was pressed) marks them as jump-start state. `$000E` written at frame 491 gives the exact death frame. `$0016` written at frame 396 gives the Goomba spawn frame.
- **"Not written during this step" is useful too.** It showed that Mario's direction bytes were frozen during the mushroom growth animation, which explained an experiment that looked broken.

**Problems and wishes**

1. **Bug: `step --png` saves a stale frame.** Every `step` PNG was byte-identical to `s0.png` (same MD5). I had to replay the whole input script with `session new` to get a screenshot, which costs time that grows with game length.
2. **Bug: `--why` gives wrong answers in some longer steps.** In a 200-frame running step from `mr.pkl` (`far.log`, `farW.log`), every address was reported as written at frame 818 by instructions that cannot write memory (`$0087` by `BEQ`, `$0009` by `INY`, `$0086` by `LDA #imm`). The same 200 frames split into two 100-frame steps gave sensible answers (`$0009` by `$8119 INC` at frame 979). 150- and 260-frame steps elsewhere were fine, so it probably depends on context length in tokens, not frames. The tool should at least warn when the writer is not a store or read-modify-write instruction.
3. `--why` only returns the *last* writer. I wanted all writers in the step, or the first write after a frame (for example, "who wrote `$0754` first"), and the value before the write.
4. `session ram` with several single addresses prints one line per address with its row offset, which is hard to read. A `name=value` list or a built-in diff (`session diff a.pkl b.pkl`) would save the helper script I wrote (`diff.py`).
5. No way to write RAM (for example, set `$075F`/`$0760`), so world/level could not be verified. A `session poke` would make hypotheses testable directly.
6. A per-frame trace of a few addresses (`--watch 0086,00CE`) would show trajectories (jump arc, speed ramp) without many short steps.
