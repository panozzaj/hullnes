"""nesformer's scanline renderer (third_party/nesformer/host.py: Host.render_line), compiled with
numba. Same logic, line for line; the board's CHR banking and nametable mirroring arrive as
1 KB page tables, which covers every board here (MMC3 banks CHR in 1 KB units, the others in
larger ones). tests/test_hullnes.py checks it against the Python version pixel for pixel."""

import numpy as np
from numba import njit


def page_tables(board):
    """CHR offsets of the eight 1 KB pages of $0000-$1FFF, and VRAM offsets of the four
    nametables, under the board's current banking and mirroring."""
    chr_pages = np.array([board.chr_index(i * 0x400) for i in range(8)], np.int64)
    nt_pages = np.array([board.nt_index(0x2000 + i * 0x400) for i in range(4)], np.int64)
    return chr_pages, nt_pages


@njit(cache=True)
def render_line(line, row, pal, vram, oam, chr_data, chr_pages, nt_pages, mask, ctrl, x, v, t):
    """Draws one scanline into row. Returns (new v, status bits to set: 0x20 overflow,
    0x40 sprite-0 hit)."""
    status = 0
    if not mask & 0x18:
        bgc = pal[0] & 0x3F
        for i in range(256):
            row[i] = bgc
        return v, status
    show_bg, show_sp = mask & 0x08, mask & 0x10
    bg = np.zeros(256, np.int64)
    bgp = np.zeros(256, np.int64)
    if show_bg:
        vv = v
        fine_y = vv >> 12 & 7
        base = (ctrl & 0x10) << 8
        px = -x
        for _ in range(33):
            nt = vv & 0x0FFF
            tile = vram[nt_pages[nt >> 10] + (nt & 0x3FF)]
            at_a = (0x03C0 | (vv & 0x0C00) | (vv >> 4 & 0x38) | (vv >> 2 & 7)) & 0x0FFF
            at = vram[nt_pages[at_a >> 10] + (at_a & 0x3FF)]
            shift = (vv >> 4 & 4) | (vv & 2)
            palno = at >> shift & 3
            a = base + tile * 16 + fine_y
            lo = chr_data[chr_pages[a >> 10] + (a & 0x3FF)]
            a8 = a + 8
            hi = chr_data[chr_pages[a8 >> 10] + (a8 & 0x3FF)]
            for b in range(8):
                xx = px + b
                if 0 <= xx < 256:
                    bg[xx] = (lo >> (7 - b) & 1) | (hi >> (7 - b) & 1) << 1
                    bgp[xx] = palno
            px += 8
            if (vv & 0x1F) == 31:
                vv = (vv & ~0x1F) ^ 0x0400
            else:
                vv += 1
        if not mask & 0x02:
            for i in range(8):
                bg[i] = 0
    sp = np.zeros(256, np.int64)
    spp = np.zeros(256, np.int64)
    spfront = np.zeros(256, np.int64)
    if show_sp:
        h = 16 if ctrl & 0x20 else 8
        found = 0
        hit0 = False
        for n in range(64):
            y = oam[4 * n]
            r = line - (y + 1)
            if not (0 <= r < h):
                continue
            found += 1
            if found > 8:
                status |= 0x20
                break
            tile, attr, x0 = oam[4 * n + 1], oam[4 * n + 2], oam[4 * n + 3]
            if attr & 0x80:
                r = h - 1 - r
            if h == 8:
                a = ((ctrl & 0x08) << 9) + tile * 16 + r
            else:
                a = (tile & 1) * 0x1000 + (tile & 0xFE) * 16 + (r & 7) + (16 if r >= 8 else 0)
            lo = chr_data[chr_pages[a >> 10] + (a & 0x3FF)]
            a8 = a + 8
            hi = chr_data[chr_pages[a8 >> 10] + (a8 & 0x3FF)]
            for b in range(8):
                xx = x0 + b
                if xx > 255:
                    break
                bit = b if attr & 0x40 else 7 - b
                c = (lo >> bit & 1) | (hi >> bit & 1) << 1
                if not c or sp[xx]:
                    continue
                if xx < 8 and not mask & 0x04:
                    continue
                sp[xx] = c
                spp[xx] = attr & 3
                spfront[xx] = 0 if attr & 0x20 else 1
                if n == 0 and bg[xx] and xx != 255:
                    hit0 = True
        if hit0 and show_bg:
            status |= 0x40
    for i in range(256):
        if sp[i] and (spfront[i] or not bg[i]):
            row[i] = pal[0x10 | spp[i] << 2 | sp[i]] & 0x3F
        elif bg[i]:
            row[i] = pal[bgp[i] << 2 | bg[i]] & 0x3F
        else:
            row[i] = pal[0] & 0x3F
    # end of the line: fine Y++ (dot 256), horizontal bits from t (dot 257)
    if (v & 0x7000) != 0x7000:
        v += 0x1000
    else:
        v &= ~0x7000
        cy = (v & 0x03E0) >> 5
        if cy == 29:
            cy = 0
            v ^= 0x0800
        elif cy == 31:
            cy = 0
        else:
            cy += 1
        v = (v & ~0x03E0) | cy << 5
    return (v & ~0x041F) | (t & 0x041F), status
