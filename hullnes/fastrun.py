"""Free-running with the host compiled too: one numba loop runs the transformer step and the
host's per-token bookkeeping (cycles, PPU registers, scanline events, idle-loop skip, the mapper
board), returning to Python only at frame ends, context rollovers, HALT, or a token whose
embedding has not been computed yet.

This is a port of the Python host (third_party/nesformer/host.py plus hullnes/host.py's
BoardHost) and of hullnes/mapper.py's boards, statement for statement. run.py keeps the Python
path; tests/test_hullnes.py requires both to produce the same tokens and the same frames.
"""

from types import SimpleNamespace

import numpy as np
from numba import njit

from hullnes import tokens as T
from hullnes.engine import _step
from hullnes.ppu import render_line
from nesformer import spec as S
from nesformer.cpu import LENGTH, OPS
from nesformer.host import EVENTS, FRAME_DOTS

# --- token decoding tables -----------------------------------------------------------------------
FAMS = list(S.FAMILIES) + S.CONTROL_NAMES + ["RB", "RI", "Z"]
_F = {n: i for i, n in enumerate(FAMS)}
F_A, F_X, F_Y, F_ST, F_RM, F_P, F_S, F_NT = (_F[n] for n in ("A", "X", "Y", "ST", "RM", "P", "S",
                                                             "NT"))
F_J, F_JI, F_R, F_PSH, F_IOR, F_IN, F_IOW, F_DM = (_F[n] for n in ("J", "JI", "R", "PSH", "IOR",
                                                                   "IN", "IOW", "DM"))
F_MAP, F_OPF, F_JL, F_RL, F_JIL = (_F[n] for n in ("MAP", "OPF", "JL", "RL", "JIL"))
F_NMI, F_HALT, F_GO, F_IRQ = (_F[n] for n in ("NMI", "HALT", "GO", "IRQ"))
TOK_NMI, TOK_IRQ, TOK_HALT = S.tok("NMI"), S.tok("IRQ"), S.tok("HALT")
IN_BASE, IOR_BASE = S.FAMILIES["IN"][0], S.FAMILIES["IOR"][0]


def token_tables():
    fam = np.full(T.VOCAB, -1, np.int64)
    pay = np.zeros(T.VOCAB, np.int64)
    for name, (base, size) in S.FAMILIES.items():
        fam[base:base + size] = _F[name]
        pay[base:base + size] = np.arange(size)
    for name, t in S.CONTROLS.items():
        fam[t] = _F[name]
    fam[T.RB_BASE:T.RB_BASE + 256] = _F["RB"]
    fam[T.RI_BASE:T.RI_BASE + 256] = _F["RI"]
    fam[T.Z_TOK] = _F["Z"]
    return fam, pay


OP_CYC = np.zeros(256, np.int64)
OP_REL = np.zeros(256, np.int64)
for _op, (_mn, _mode, _cyc) in OPS.items():
    OP_CYC[_op], OP_REL[_op] = _cyc, int(_mode == "rel")
EV_DOT = np.array([e[0] for e in EVENTS], np.int64)
EV_ACT = np.array([{"render": 0, "vblank": 1, "prerender": 2, "frame": 3}[e[1]] for e in EVENTS],
                  np.int64)
EV_LINE = np.array([e[2] for e in EVENTS], np.int64)
N_EVENTS = len(EVENTS)


@njit(cache=True, inline="always")
def length_of(f, p):
    if f == F_A or f == F_X or f == F_Y:
        return p >> 11
    if f == F_S or f == F_PSH:
        return p >> 8
    if f == F_ST:
        return p >> 9
    if f == F_RM:
        return p >> 10
    if f == F_P:
        return p >> 7
    if f == F_NT:
        return p
    if f == F_IOW:
        return p >> 14
    if f == F_MAP:
        return p >> 14
    return 0


@njit(cache=True, inline="always")
def is_final(f, p):
    return (f == F_J or f == F_JI or f == F_R or f == F_JL or f == F_RL or f == F_JIL
            or f == F_HALT or f == F_GO or length_of(f, p) > 0)


@njit(cache=True, inline="always")
def anchor_addr(f, p):
    return p if (f == F_JL or f == F_RL or f == F_JIL) else 0x8000 + p


# --- the board (hullnes/mapper.py) ---------------------------------------------------------------
# bd: [number, n16, n8, chr_size, vertical, chr_ram, regs...]
B_NUM, B_N16, B_N8, B_CHRSZ, B_VERT, B_CHRRAM, B_R = 0, 1, 2, 3, 4, 5, 6
# UxROM/CNROM: R+0 bank. MMC1: R+0..3 regs, R+4 shift, R+5 count.
# MMC3: R+0..7 regs, R+8 select, R+9 horizontal, R+10 latch, R+11 counter, R+12 reload,
# R+13 enabled, R+14 irq_pending
M3_SEL, M3_HOR, M3_LATCH, M3_CNT, M3_RELOAD, M3_EN, M3_PEND = (B_R + 8 + i for i in range(7))


def board_array(board, chr_ram: bool) -> np.ndarray:
    bd = np.zeros(B_R + 16, np.int64)
    bd[B_NUM], bd[B_N16] = board.number, len(board.prg) // 0x4000
    bd[B_N8], bd[B_CHRSZ] = len(board.prg) // 0x2000, board.chr_size
    bd[B_VERT], bd[B_CHRRAM] = int(board.vertical), int(chr_ram)
    if board.number in (2, 3):
        bd[B_R] = board.bank
    elif board.number == 1:
        bd[B_R:B_R + 4] = board.regs
        bd[B_R + 4], bd[B_R + 5] = board.shift, board.count
    elif board.number == 4:
        bd[B_R:B_R + 8] = board.regs
        bd[M3_SEL], bd[M3_HOR], bd[M3_LATCH] = board.select, board.horizontal, board.latch
        bd[M3_CNT], bd[M3_RELOAD] = board.counter, int(board.reload)
        bd[M3_EN], bd[M3_PEND] = int(board.enabled), int(board.irq_pending)
    return bd


@njit(cache=True)
def board_apply(bd, p):
    n = bd[B_NUM]
    if n == 2 or n == 3:
        bd[B_R] = p & 255
    elif n == 1:
        val5, count, commit, reg = p & 31, p >> 5 & 7, p >> 8 & 1, p >> 9 & 3
        if commit:
            bd[B_R + reg] = val5
            bd[B_R + 4] = 0
            bd[B_R + 5] = 0
        else:
            bd[B_R + 4] = val5
            bd[B_R + 5] = count
    elif n == 4:
        v, kind, target = p & 255, p >> 8 & 7, p >> 11 & 7
        if kind == 0:
            bd[M3_SEL] = v
        elif kind == 1:
            bd[B_R + target] = v
        elif kind == 2:
            bd[M3_HOR] = v & 1
        elif kind == 4:
            bd[M3_LATCH] = v
        elif kind == 5:
            bd[M3_CNT] = 0
            bd[M3_RELOAD] = 1
        elif kind == 6:
            bd[M3_EN] = 0
            bd[M3_PEND] = 0
        elif kind == 7:
            bd[M3_EN] = 1


@njit(cache=True)
def prg_index(bd, n_prg, a):
    n = bd[B_NUM]
    n16 = bd[B_N16]
    if n == 0 or n == 3:
        return (a - 0x8000) % n_prg
    if n == 2:
        b = bd[B_R] & (n16 - 1) if a < 0xC000 else n16 - 1
        return b * 0x4000 + (a & 0x3FFF)
    if n == 1:
        mode, p = bd[B_R] >> 2 & 3, bd[B_R + 3] & 15 & (n16 - 1)
        if mode < 2:
            return (p & ~1) * 0x4000 + (a & 0x7FFF)
        if mode == 2:
            b = 0 if a < 0xC000 else p
        else:
            b = p if a < 0xC000 else n16 - 1
        return b * 0x4000 + (a & 0x3FFF)
    n8 = bd[B_N8]
    r6, r7 = bd[B_R + 6] & (n8 - 1), bd[B_R + 7] & (n8 - 1)
    slot = (a - 0x8000) >> 13
    if bd[M3_SEL] >> 6 & 1:
        b = n8 - 2 if slot == 0 else r7 if slot == 1 else r6 if slot == 2 else n8 - 1
    else:
        b = r6 if slot == 0 else r7 if slot == 1 else n8 - 2 if slot == 2 else n8 - 1
    return b * 0x2000 + (a & 0x1FFF)


@njit(cache=True)
def chr_index(bd, a):
    n = bd[B_NUM]
    a &= 0x1FFF
    size = bd[B_CHRSZ]
    if n == 3:
        return (bd[B_R] * 0x2000 + a) % size
    if n == 1:
        if bd[B_R] & 0x10:
            bank = bd[B_R + 1] if a < 0x1000 else bd[B_R + 2]
            return (bank * 0x1000 + (a & 0x0FFF)) % size
        return ((bd[B_R + 1] & ~1) * 0x1000 + a) % size
    if n == 4:
        if bd[M3_SEL] & 0x80:
            a ^= 0x1000
        if a < 0x1000:
            bank = (bd[B_R + (a >> 11)] & 0xFE) * 0x400 + (a & 0x7FF)
        else:
            bank = bd[B_R + 2 + ((a - 0x1000) >> 10)] * 0x400 + (a & 0x3FF)
        return bank % size
    return a


@njit(cache=True)
def nt_index(bd, a):
    a &= 0x0FFF
    n = bd[B_NUM]
    if n == 1:
        m = bd[B_R] & 3
        if m == 0:
            return a & 0x03FF
        if m == 1:
            return 0x400 | (a & 0x03FF)
        if m == 2:
            return a & 0x07FF
        return (a & 0x03FF) | (a & 0x0800) >> 1
    vertical = (bd[M3_HOR] == 0) if n == 4 else bd[B_VERT] != 0
    if vertical:
        return a & 0x07FF
    return (a & 0x03FF) | (a & 0x0800) >> 1


@njit(cache=True)
def mmc3_scanline(bd):
    if bd[M3_CNT] == 0 or bd[M3_RELOAD]:
        bd[M3_CNT] = bd[M3_LATCH]
        bd[M3_RELOAD] = 0
    else:
        bd[M3_CNT] -= 1
    if bd[M3_CNT] == 0 and bd[M3_EN]:
        bd[M3_PEND] = 1


# --- the host ---------------------------------------------------------------------------------
(H_CYC, H_PC, H_FRAME, H_CTRL, H_MASK, H_STATUS, H_OAMADDR, H_V, H_T, H_X, H_W, H_BUF,
 H_FBASE, H_EVI, H_NMIP, H_SHIFT, H_STROBE, H_RECSTART, H_RECCYC, H_RECOP, H_INNMI, H_HALTED,
 H_RECDMA, H_PREVTGT, H_PREVLEN, H_ITERLEN, H_ITERCYC0, H_SKCYC, H_SKITER, H_EVSEQ, H_ITEREV0,
 H_FLAGI, H_SKIP, H_BUTTONS, H_NFRAMES_OUT) = range(35)
H_SIZE = 35
ITER_CAP = 8192


@njit(cache=True)
def pal_index(a):
    a &= 0x1F
    if a == 0x10 or a == 0x14 or a == 0x18 or a == 0x1C:
        a -= 0x10
    return a


@njit(cache=True)
def catch_up(H, bd, target, fbuf, frames_out, pal, vram, oam, chr_data):
    chr_pages = np.empty(8, np.int64)
    nt_pages = np.empty(4, np.int64)
    while True:
        i = H[H_EVI]
        if H[H_FBASE] + EV_DOT[i] > target:
            return
        act = EV_ACT[i]
        if act == 0:
            line = EV_LINE[i]
            for k in range(8):
                chr_pages[k] = chr_index(bd, k * 0x400)
            for k in range(4):
                nt_pages[k] = nt_index(bd, 0x2000 + k * 0x400)
            v, st = render_line(line, fbuf[line], pal, vram, oam, chr_data, chr_pages, nt_pages,
                                H[H_MASK], H[H_CTRL], H[H_X], H[H_V], H[H_T])
            H[H_V] = v
            H[H_STATUS] |= st
            if bd[B_NUM] == 4 and H[H_MASK] & 0x18:
                if line == 0:
                    mmc3_scanline(bd)
                mmc3_scanline(bd)
        elif act == 1:
            H[H_STATUS] |= 0x80
            if H[H_CTRL] & 0x80:
                H[H_NMIP] = 1
        elif act == 2:
            H[H_STATUS] &= 0x1F
            if H[H_MASK] & 0x18:
                H[H_V] = (H[H_V] & ~0x7BE0) | (H[H_T] & 0x7BE0)
        H[H_EVI] += 1
        H[H_EVSEQ] += 1
        if act == 3:
            n = H[H_NFRAMES_OUT]
            frames_out[n, :, :] = fbuf
            H[H_NFRAMES_OUT] = n + 1
            H[H_FRAME] += 1
            H[H_FBASE] += FRAME_DOTS
            H[H_EVI] = 0


@njit(cache=True)
def access_cycle(H):
    if H[H_RECOP] < 0:
        return H[H_RECCYC]
    return H[H_RECCYC] + OP_CYC[H[H_RECOP]] - 1


@njit(cache=True)
def vread(H, bd, a, vram, pal, chr_data):
    a &= 0x3FFF
    if a < 0x2000:
        return chr_data[chr_index(bd, a)]
    if a < 0x3F00:
        return vram[nt_index(bd, a)]
    return pal[pal_index(a)]


@njit(cache=True)
def io_read(H, bd, reg, fbuf, frames_out, pal, vram, oam, chr_data):
    catch_up(H, bd, access_cycle(H) * 3, fbuf, frames_out, pal, vram, oam, chr_data)
    if 32 <= reg < 40:
        r = reg - 32
        if r == 2:
            v = H[H_STATUS] | (H[H_BUF] & 0x1F)
            H[H_STATUS] &= 0x7F
            H[H_W] = 0
            return v
        if r == 4:
            return oam[H[H_OAMADDR]]
        if r == 7:
            a = H[H_V] & 0x3FFF
            v = H[H_BUF]
            H[H_BUF] = vread(H, bd, a, vram, pal, chr_data)
            if a >= 0x3F00:
                v = pal[pal_index(a)]
                H[H_BUF] = vram[nt_index(bd, a - 0x1000)]
            H[H_V] = (H[H_V] + (32 if H[H_CTRL] & 4 else 1)) & 0x7FFF
            return v
        return H[H_BUF] & 0xFF
    if reg == 0x16:
        bit = H[H_SHIFT] & 1
        if not H[H_STROBE]:
            H[H_SHIFT] = (H[H_SHIFT] >> 1) | 0x80
        return 0x40 | bit
    if reg == 0x17:
        return 0x40
    return 0


@njit(cache=True)
def io_write(H, bd, reg, v, fbuf, frames_out, pal, vram, oam, chr_data):
    catch_up(H, bd, access_cycle(H) * 3, fbuf, frames_out, pal, vram, oam, chr_data)
    if 32 <= reg < 40:
        r = reg - 32
        if r == 0:
            if v & 0x80 and not H[H_CTRL] & 0x80 and H[H_STATUS] & 0x80:
                H[H_NMIP] = 1
            H[H_CTRL] = v
            H[H_T] = (H[H_T] & ~0x0C00) | (v & 3) << 10
        elif r == 1:
            H[H_MASK] = v
        elif r == 3:
            H[H_OAMADDR] = v
        elif r == 4:
            oam[H[H_OAMADDR]] = v
            H[H_OAMADDR] = (H[H_OAMADDR] + 1) & 255
        elif r == 5:
            if H[H_W] == 0:
                H[H_T] = (H[H_T] & ~0x1F) | v >> 3
                H[H_X] = v & 7
            else:
                H[H_T] = (H[H_T] & ~0x73E0) | (v & 7) << 12 | (v & 0xF8) << 2
            H[H_W] ^= 1
        elif r == 6:
            if H[H_W] == 0:
                H[H_T] = (H[H_T] & 0x00FF) | (v & 0x3F) << 8
            else:
                H[H_T] = (H[H_T] & 0x7F00) | v
                H[H_V] = H[H_T]
            H[H_W] ^= 1
        elif r == 7:
            a = H[H_V] & 0x3FFF
            if a < 0x2000:
                if bd[B_CHRRAM]:
                    chr_data[chr_index(bd, a)] = v
            elif a < 0x3F00:
                vram[nt_index(bd, a)] = v
            else:
                pal[pal_index(a)] = v & 0x3F
            H[H_V] = (H[H_V] + (32 if H[H_CTRL] & 4 else 1)) & 0x7FFF
    elif reg == 0x16:
        H[H_STROBE] = v & 1
        if H[H_STROBE]:
            H[H_SHIFT] = H[H_BUTTONS]


@njit(cache=True)
def skip_cap(H, bd):
    if bd[B_NUM] != 4 or not bd[M3_EN] or not H[H_MASK] & 0x18:
        return -1
    counter, reload, i, fb = bd[M3_CNT], bd[M3_RELOAD], H[H_EVI], H[H_FBASE]
    for _ in range(2 * N_EVENTS):
        if EV_ACT[i] == 0:
            clocks = 2 if EV_LINE[i] == 0 else 1
            for _c in range(clocks):
                if counter == 0 or reload:
                    counter, reload = bd[M3_LATCH], 0
                else:
                    counter -= 1
                if counter == 0:
                    return fb + EV_DOT[i]
        act = EV_ACT[i]
        i += 1
        if act == 3:
            i, fb = 0, fb + FRAME_DOTS
    return -1


@njit(cache=True)
def maybe_skip(H, bd, target, iter_toks, prev_body, fam, pay, fbuf, frames_out, pal, vram, oam,
               chr_data):
    n = H[H_ITERLEN]
    period = H[H_CYC] - H[H_ITERCYC0]
    same = H[H_PREVLEN] == n and target == H[H_PREVTGT] and n <= ITER_CAP
    if same:
        for k in range(n):
            if prev_body[k] != iter_toks[k]:
                same = False
                break
    quiet = H[H_EVSEQ] == H[H_ITEREV0] and not H[H_NMIP]
    if n <= ITER_CAP:
        prev_body[:n] = iter_toks[:n]
        H[H_PREVLEN] = n
    else:
        H[H_PREVLEN] = -1
    H[H_PREVTGT] = target
    H[H_ITERLEN] = 0
    H[H_ITERCYC0] = H[H_CYC]
    H[H_ITEREV0] = H[H_EVSEQ]
    if not (H[H_SKIP] and same and quiet and period > 0):
        return
    reads = False
    for k in range(n):
        f = fam[iter_toks[k]]
        if not (f == F_P or f == F_A or f == F_X or f == F_Y or f == F_NT or f == F_J
                or f == F_LO_ or f == F_HI_ or f == F_IOR or f == F_IN):
            return
        if f == F_IOR:
            if pay[iter_toks[k]] != 34:
                return
            reads = True
    if reads:
        ev_dot = H[H_FBASE] + EV_DOT[H[H_EVI]]
    elif H[H_CTRL] & 0x80:
        ev_dot = H[H_FBASE] + 241 * 341 + 1
        if H[H_EVI] > 240:
            ev_dot += FRAME_DOTS
    else:
        return
    cap = skip_cap(H, bd)
    if cap >= 0 and cap < ev_dot:
        ev_dot = cap
    k = (-(-ev_dot // 3) - H[H_CYC]) // period - 1
    if k > 0:
        H[H_CYC] += k * period
        H[H_ITERCYC0] = H[H_CYC]
        H[H_SKCYC] += k * period
        H[H_SKITER] += k
        catch_up(H, bd, H[H_CYC] * 3, fbuf, frames_out, pal, vram, oam, chr_data)
        H[H_ITEREV0] = H[H_EVSEQ]


F_LO_, F_HI_ = _F["LO"], _F["HI"]


@njit(cache=True)
def feed(H, bd, t, fam, pay, prg, n_prg, iter_toks, prev_body, fbuf, frames_out, pal, vram, oam,
         chr_data):
    f, p = fam[t], pay[t]
    if f == F_GO:
        return
    n = H[H_ITERLEN]
    if n < ITER_CAP:
        iter_toks[n] = t
    H[H_ITERLEN] = n + 1
    if H[H_RECSTART]:
        H[H_RECSTART] = 0
        H[H_RECDMA] = 0
        H[H_RECCYC] = H[H_CYC]
        H[H_INNMI] = 1 if (f == F_NMI or f == F_IRQ) else 0
        if f == F_NMI or f == F_IRQ or f == F_R or f == F_RL:
            H[H_RECOP] = -1
        elif f == F_OPF:
            H[H_RECOP] = p & 255
        else:
            H[H_RECOP] = prg[prg_index(bd, n_prg, 0x8000 | (H[H_PC] & 0x7FFF))]
    if f == F_IOW:
        io_write(H, bd, p >> 8 & 63, p & 255, fbuf, frames_out, pal, vram, oam, chr_data)
        if (p >> 8 & 63) == 0x14:
            H[H_RECDMA] = 1
    elif f == F_DM:
        oam[(H[H_OAMADDR] + (p >> 8)) & 255] = p & 255
    elif f == F_HALT:
        H[H_HALTED] = 1
    if is_final(f, p):
        # finish
        H[H_RECSTART] = 1
        if f == F_R or f == F_RL:
            H[H_PC] = anchor_addr(f, p)
        elif f != F_HALT:
            if H[H_INNMI]:
                cyc = 7
            else:
                op = H[H_RECOP]
                cyc = OP_CYC[op]
                if f == F_A or f == F_X or f == F_Y:
                    cyc += p >> 10 & 1
                elif f == F_P:
                    cyc += p >> 6 & 1
                if H[H_RECDMA]:
                    cyc += 513
                if OP_REL[op] and (f == F_J or f == F_JL):
                    cyc += 1 + (1 if ((H[H_PC] + 2) & 0xFF00) != (anchor_addr(f, p) & 0xFF00)
                                else 0)
            if f == F_J or f == F_JI or f == F_JL or f == F_JIL:
                H[H_PC] = anchor_addr(f, p)
            else:
                H[H_PC] = (H[H_PC] + length_of(f, p)) & 0xFFFF
            H[H_CYC] += cyc
            catch_up(H, bd, H[H_CYC] * 3, fbuf, frames_out, pal, vram, oam, chr_data)
            if (f == F_J or f == F_JL) and not H[H_INNMI]:
                maybe_skip(H, bd, anchor_addr(f, p), iter_toks, prev_body, fam, pay, fbuf,
                           frames_out, pal, vram, oam, chr_data)
            elif f == F_JI or f == F_JIL or f == F_R or f == F_RL:
                H[H_PREVLEN] = -1
                H[H_ITERLEN] = 0
                H[H_ITERCYC0] = H[H_CYC]
                H[H_ITEREV0] = H[H_EVSEQ]
    # BoardHost: board writes take effect after the record's opcode fetch; follow I
    if f == F_MAP:
        board_apply(bd, p & 0x3FFF)
    elif f == F_P:
        H[H_FLAGI] = p >> 2 & 1
    elif f == F_JI or f == F_JIL:
        H[H_FLAGI] = 1


# --- the loop --------------------------------------------------------------------------------
R_DONE, R_FRAME, R_ROLL, R_EMB, R_FULL, R_ERR = 0, 1, 2, 3, 4, 5


@njit(cache=True)
def run(H, bd, fam, pay, prg, n_prg, iter_toks, prev_body, fbuf, frames_out, pal, vram, oam,
        chr_data, toks, ntok, max_tokens, frames_target, pred_in, roll_at,
        emb_start, emb_len, emb_dim, emb_val, eng_v, pos_in, d_pos, d_pos2,
        e_ptr, e_dim, e_coef, sched, h_q, h_kx, h_ky, h_lo, h_hi, h_vptr, h_vform, h_out,
        n_m, n_b, n_l, n_r, n_vals, n_count, h_base, h_vbase, tmp, s_form, s_out, sums,
        u_a, u_aconst, u_base, u_bc, b_form, bval, u_c, u_out, g_base, g_out, g_off, g_n, g_c,
        g_w, g_wc, out_form, r_form, r_lo, r_hi, state):
    """state: [pred, pos, ntok, missing token]. Returns a reason code."""
    pred, pos = pred_in, pos_in
    batch = np.empty(2, np.int64)
    while True:
        state[0], state[1], state[2] = pred, pos, ntok
        if H[H_FRAME] >= frames_target or H[H_HALTED]:
            return R_DONE
        if max_tokens >= 0 and ntok >= max_tokens:
            return R_DONE
        if H[H_NFRAMES_OUT] > 0:
            return R_FRAME
        if ntok + 2 > toks.shape[0]:
            return R_FULL
        if H[H_NMIP] and H[H_RECSTART] and not H[H_HALTED]:
            H[H_NMIP] = 0
            t = TOK_NMI
        elif bd[B_NUM] == 4 and bd[M3_PEND] and H[H_RECSTART] and not H[H_HALTED] \
                and not H[H_FLAGI]:
            t = TOK_IRQ
        else:
            t = pred
            if t < 0 or t >= emb_start.shape[0]:
                return R_ERR
            if emb_start[t] < 0:
                state[3] = t
                return R_EMB
        nb = 1
        batch[0] = t
        feed(H, bd, t, fam, pay, prg, n_prg, iter_toks, prev_body, fbuf, frames_out, pal, vram,
             oam, chr_data)
        if fam[t] == F_IOR:
            v = io_read(H, bd, pay[t], fbuf, frames_out, pal, vram, oam, chr_data)
            batch[1] = IN_BASE + v
            nb = 2
            feed(H, bd, batch[1], fam, pay, prg, n_prg, iter_toks, prev_body, fbuf, frames_out,
                 pal, vram, oam, chr_data)
        for k in range(nb):
            tt = batch[k]
            toks[ntok] = tt
            ntok += 1
            s0 = emb_start[tt]
            y = _step(eng_v, emb_dim[s0:s0 + emb_len[tt]], emb_val[s0:s0 + emb_len[tt]], pos,
                      d_pos, d_pos2, e_ptr, e_dim, e_coef, sched, h_q, h_kx, h_ky, h_lo, h_hi,
                      h_vptr, h_vform, h_out, n_m, n_b, n_l, n_r, n_vals, n_count, h_base,
                      h_vbase, tmp, s_form, s_out, sums, u_a, u_aconst, u_base, u_bc, b_form,
                      bval, u_c, u_out, g_base, g_out, g_off, g_n, g_c, g_w, g_wc, out_form,
                      r_form, r_lo, r_hi, False)
            pos += 1
            if y < 0:
                state[0], state[1], state[2] = -1, pos, ntok
                return R_ERR
            pred = np.int64(np.rint(y))
        if t == TOK_HALT:
            state[0], state[1], state[2] = pred, pos, ntok
            return R_DONE
        f, p = fam[toks[ntok - 1]], pay[toks[ntok - 1]]
        if pos > roll_at and is_final(f, p):
            state[0], state[1], state[2] = pred, pos, ntok
            return R_ROLL


# --- Python side -------------------------------------------------------------------------------
_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = token_tables()
    return _TABLES


class EmbTable:
    """The engine's token embeddings as a flat table numba can index, filled on demand."""

    def __init__(self, eng):
        self.eng = eng
        self.start = np.full(T.VOCAB, -1, np.int64)
        self.length = np.zeros(T.VOCAB, np.int64)
        self.dim = np.zeros(1 << 16, np.int64)
        self.val = np.zeros(1 << 16, np.float64)
        self.n = 0
        for t in [TOK_NMI, TOK_IRQ, *range(IN_BASE, IN_BASE + 256)]:
            self.add(t)

    def add(self, t: int) -> None:
        if self.start[t] >= 0:
            return
        d, v = self.eng.embed(t)
        while self.n + len(d) > len(self.dim):
            self.dim = np.concatenate([self.dim, np.zeros_like(self.dim)])
            self.val = np.concatenate([self.val, np.zeros_like(self.val)])
        self.dim[self.n:self.n + len(d)] = d
        self.val[self.n:self.n + len(d)] = v
        self.start[t], self.length[t] = self.n, len(d)
        self.n += len(d)


def host_array(host) -> np.ndarray:
    H = np.zeros(H_SIZE, np.int64)
    H[H_CYC], H[H_PC], H[H_FRAME] = host.cycles, host.pc, host.frame
    H[H_CTRL], H[H_MASK], H[H_STATUS], H[H_OAMADDR] = host.ctrl, host.mask, host.status, \
        host.oamaddr
    H[H_V], H[H_T], H[H_X], H[H_W], H[H_BUF] = host.v, host.t, host.x, host.w, host.buf
    H[H_FBASE], H[H_EVI], H[H_NMIP] = host.frame_base, host.ev_i, int(host.nmi_pending)
    H[H_SHIFT], H[H_STROBE], H[H_RECSTART] = host.shift, host.strobe, int(host.rec_start)
    H[H_RECCYC] = host.rec_cycles
    H[H_RECOP] = -1 if host.rec_op is None else host.rec_op
    H[H_INNMI], H[H_HALTED] = int(host.in_nmi), int(host.halted)
    H[H_RECDMA] = int(getattr(host, "rec_dma", False))
    H[H_PREVLEN] = -1 if host.prev_body is None else len(host.prev_body)
    H[H_PREVTGT] = -1 if host.prev_target is None else host.prev_target
    H[H_ITERLEN], H[H_ITERCYC0] = len(host.iter_toks), host.iter_cycle0
    H[H_SKCYC], H[H_SKITER], H[H_EVSEQ], H[H_ITEREV0] = host.skipped_cycles, host.skipped_iters, \
        host.ev_seq, host.iter_ev0
    H[H_FLAGI], H[H_SKIP] = host.flag_i, int(host.skip)
    return H


def run_fast(ses, host, pred: int, toks: list[int], frames: int, max_tokens, buttons,
             keep_frames: bool, on_frame, roll_margin: int):
    """Continue a free run in compiled code from a fresh BoardHost and the prediction after GO.
    Returns (trace, result) where result has frames, frame, halted, rollovers."""
    eng = ses.eng
    cp = eng.cp
    fam, pay = tables()
    emb = EmbTable(eng)
    for t in toks:
        emb.add(t)
    H = host_array(host)
    bd = board_array(host.board, host.cart.chr_ram)
    prg = np.frombuffer(host.board.prg, np.uint8)
    n_prg = len(host.board.prg)
    iter_toks = np.zeros(ITER_CAP + 1, np.int64)
    prev_body = np.zeros(ITER_CAP + 1, np.int64)
    if host.prev_body is not None:
        prev_body[:len(host.prev_body)] = host.prev_body
    iter_toks[:len(host.iter_toks)] = host.iter_toks
    fbuf = np.zeros((240, 256), np.uint8)
    frames_out = np.zeros((4, 240, 256), np.uint8)
    pal = np.frombuffer(host.pal, np.uint8).copy()
    vram = np.frombuffer(host.vram, np.uint8).copy()
    oam = np.frombuffer(host.oam, np.uint8).copy()
    chr_data = np.frombuffer(host.cart.chr.data, np.uint8).copy()
    out = np.zeros(1 << 20, np.int64)
    out[:len(toks)] = toks
    ntok = len(toks)
    state = np.zeros(4, np.int64)
    result = SimpleNamespace(frames=[], frame=host.frame, halted=False, rollovers=0, cycles=0)
    seen = host.frame

    def mask():
        if buttons is None:
            return 0
        return buttons(int(H[H_FRAME])) if callable(buttons) else int(buttons)

    while True:
        H[H_BUTTONS] = mask()
        mt = -1 if max_tokens is None else max_tokens
        code = run(H, bd, fam, pay, prg, n_prg, iter_toks, prev_body, fbuf, frames_out, pal,
                   vram, oam, chr_data, out, ntok, mt, frames, pred,
                   eng.max_positions - roll_margin, emb.start, emb.length, emb.dim, emb.val,
                   eng.v, eng.pos, cp.d_pos, cp.d_pos2, cp.e_ptr, cp.e_dim, cp.e_coef, cp.sched,
                   cp.h_q, cp.h_kx, cp.h_ky, cp.h_lo, cp.h_hi, cp.h_vptr, cp.h_vform, cp.h_out,
                   eng.n_m, eng.n_b, eng.n_l, eng.n_r, eng.n_vals, eng.n_count, eng.h_base,
                   eng.h_vbase, eng.tmp, cp.s_form, cp.s_out, eng.sums, cp.u_a, cp.u_aconst,
                   cp.u_base, cp.u_bc, cp.b_form, eng.bval, cp.u_c, cp.u_out, cp.g_base,
                   cp.g_out, cp.g_off, cp.g_n, cp.g_c, cp.g_w, cp.g_wc, cp.out_form, cp.r_form,
                   cp.r_lo, cp.r_hi, state)
        pred, eng.pos, ntok = int(state[0]), int(state[1]), int(state[2])
        n = int(H[H_NFRAMES_OUT])
        for k in range(n):
            if keep_frames:
                result.frames.append([bytes(r) for r in frames_out[k]])
        H[H_NFRAMES_OUT] = 0
        if n and on_frame and H[H_FRAME] > seen:
            seen = int(H[H_FRAME])
            result.frame = seen
            on_frame(result)
        if code == R_DONE:
            break
        if code == R_EMB:
            emb.add(int(state[3]))
        elif code == R_FULL:
            out = np.concatenate([out, np.zeros_like(out)])
        elif code == R_ROLL:
            pred = ses.rollover()
            result.rollovers += 1
        elif code == R_ERR:
            raise ValueError(f"engine error at position {eng.pos}")
    result.frame = int(H[H_FRAME])
    result.halted = bool(H[H_HALTED])
    result.cycles = int(H[H_CYC])
    result.skipped_iters = int(H[H_SKITER])
    return out[:ntok].tolist(), result
