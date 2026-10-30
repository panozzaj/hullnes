"""Decode a circuit graph one token at a time with an exact O(log n) attention cache.

This is the transformer's forward pass for one position, written out sparsely:

  embedding   token features + [1, t, t^2]
  attention   each 2D hardmax head inserts its key line (kx_t, ky_t) and value vector into a
              Li Chao tree, then asks which stored line is highest at q_t. That is the exact
              argmax of q_t * kx_s + ky_s over s <= t in O(log range) instead of a scan over
              the whole prefix (Percepta's "HullKVCache", in its dual form: the upper envelope
              of lines is the convex hull of the key points seen edge-on).
  cumsum      a uniform head is a running sum
  ReGLU       coef * a * relu(b) for each unit, skipping units whose relu is 0

Every value is an integer held exactly in float64 (all magnitudes stay below 2^53); scores are
int64. Nothing here knows about the NES.
"""

import numpy as np
from numba import njit

from chipformer.dsl.graph import ONE, Cumsum, Expr, Graph, Lookup

K_ATTN, K_SUM, K_FFN = 0, 1, 2


class Compiled:
    """The graph flattened into arrays: linear forms in CSR, then a schedule of sublayers."""

    def __init__(self, g: Graph):
        self.g = g
        self.n_dims = len(g.dims)
        ptr, dim, coef = [0], [], []

        def form(e: Expr) -> int:
            for d, v in e.terms.items():
                if v != int(v):
                    raise ValueError(f"non-integer coefficient {v}")
                dim.append(d)
                coef.append(float(v))
            ptr.append(len(dim))
            return len(ptr) - 2

        nodes = sorted(g.nodes, key=lambda n: n.level)  # stable: creation order within a level
        sched = []  # (kind, first, last) item ranges per node group
        # attention heads
        h_q, h_kx, h_ky, h_lo, h_hi, h_vptr, h_vform, h_out, h_cap = [], [], [], [], [], [0], [], [], []
        # cumsums
        s_form, s_out = [], []
        # ffn units, grouped per sublayer: each unit's relu input is (shared base) + constant,
        # where the base is the input's linear form without its constant term. Units of one
        # piecewise-linear function share a base, so each base is evaluated once per sublayer.
        u_a, u_aconst, u_base, u_bc, u_c, u_out = [], [], [], [], [], []
        b_form = []  # base forms, in sublayer order
        level_bases: dict = {}  # base key -> base index, for the sublayer being built
        self.head_names = []

        def unit(cf, a, b, out):
            if cf != int(cf):
                raise ValueError(f"non-integer coefficient {cf}")
            key = tuple(sorted((k, v) for k, v in b.terms.items() if k != ONE))
            if key not in level_bases:
                level_bases[key] = len(b_form)
                b_form.append(form(Expr(dict(key))))
            u_base.append(level_bases[key])
            u_bc.append(float(b.terms.get(ONE, 0)))
            if set(a.terms) <= {ONE}:  # a constant multiplier: no dot product
                u_a.append(-1)
                u_aconst.append(float(a.terms.get(ONE, 0)))
            else:
                u_a.append(form(a))
                u_aconst.append(0.0)
            u_c.append(float(cf))
            u_out.append(out)

        for n in nodes:
            if isinstance(n, Lookup):
                h_q.append(form(n.q))
                h_kx.append(form(n.kx))
                h_ky.append(form(n.ky))
                h_lo.append(n.q_range[0])
                h_hi.append(n.q_range[1])
                for v, o in zip(n.values, n.out):
                    h_vform.append(form(v))
                    h_out.append(o)
                h_vptr.append(len(h_vform))
                h_cap.append(2 * (n.q_range[1] - n.q_range[0] + 1) + 2)
                self.head_names.append(n.name)
                sched.append((K_ATTN, len(h_q) - 1, len(h_q), n.level, 0, 0))
            elif isinstance(n, Cumsum):
                for v, o in zip(n.values, n.out):
                    s_form.append(form(v))
                    s_out.append(o)
                sched.append((K_SUM, len(s_form) - len(n.values), len(s_form), n.level, 0, 0))
            else:
                if sched and sched[-1][0] == K_FFN and sched[-1][3] == n.level:
                    kind, first, _, lvl, b0, _ = sched.pop()  # extend this sublayer's group
                else:
                    first, b0 = len(u_a), len(b_form)
                    level_bases.clear()
                for cf, a, b in n.terms:
                    unit(cf, a, b, n.out)
                sched.append((K_FFN, first, len(u_a), n.level, b0, len(b_form)))
        self.out_form = form(g.outputs["next"])
        self.ranges = []
        for e, lo, hi, name in getattr(g, "ranges", []):
            self.ranges.append((form(e), lo, hi, name))
        i64 = lambda x: np.asarray(x, np.int64)
        f64 = lambda x: np.asarray(x, np.float64)
        self.e_ptr, self.e_dim, self.e_coef = i64(ptr), i64(dim), f64(coef)
        self.h_q, self.h_kx, self.h_ky = i64(h_q), i64(h_kx), i64(h_ky)
        self.h_lo, self.h_hi = i64(h_lo), i64(h_hi)
        self.h_vptr, self.h_vform, self.h_out = i64(h_vptr), i64(h_vform), i64(h_out)
        self.h_cap0 = i64(h_cap)
        self.s_form, self.s_out = i64(s_form), i64(s_out)
        self.u_a, self.u_aconst, self.u_c, self.u_out = i64(u_a), f64(u_aconst), f64(u_c), i64(u_out)
        self.u_base, self.u_bc, self.b_form = i64(u_base), f64(u_bc), i64(b_form)
        self.sched = i64([(k, a0, a1, b0, b1) for k, a0, a1, _, b0, b1 in sched])
        self.r_form = i64([r[0] for r in self.ranges])
        self.r_lo = i64([r[1] for r in self.ranges])
        self.r_hi = i64([r[2] for r in self.ranges])
        self.pos_dims = [d for d, fn in g.pos_fns.items()]
        names = {g.dims[d].name: d for d in g.pos_fns}
        self.d_pos, self.d_pos2 = names["pos"], names["pos^2"]
        self.token_fns = list(g.token_fns.items())

    def stats(self) -> dict:
        return {"dims": self.n_dims, "heads": len(self.h_q), "ffn_units": len(self.u_a),
                "nonzero_weights": int(len(self.e_dim)), "sublayers": len(self.sched)}


@njit(cache=True)
def _dot(f, v, e_ptr, e_dim, e_coef):
    s = 0.0
    for i in range(e_ptr[f], e_ptr[f + 1]):
        s += e_coef[i] * v[e_dim[i]]
    return s


@njit(cache=True)
def _step(v, tdims, tvals, pos, d_pos, d_pos2,
          e_ptr, e_dim, e_coef, sched,
          h_q, h_kx, h_ky, h_lo, h_hi, h_vptr, h_vform, h_out,
          n_m, n_b, n_l, n_r, n_vals, n_count, h_base, h_vbase, tmp,
          s_form, s_out, sums,
          u_a, u_aconst, u_base, u_bc, b_form, bval, u_c, u_out, out_form,
          r_form, r_lo, r_hi, check):
    """One position. Returns the output value; on a range violation returns -1 - index."""
    v[:] = 0.0
    v[0] = 1.0
    for i in range(tdims.shape[0]):
        v[tdims[i]] = tvals[i]
    v[d_pos] = float(pos)
    v[d_pos2] = float(pos) * float(pos)
    for si in range(sched.shape[0]):
        kind, a0, a1 = sched[si, 0], sched[si, 1], sched[si, 2]
        if kind == 2:  # ReGLU units: shared bases first, then each unit's relu and multiplier
            b0, b1 = sched[si, 3], sched[si, 4]
            for k in range(b0, b1):
                bval[k] = _dot(b_form[k], v, e_ptr, e_dim, e_coef)
            for u in range(a0, a1):
                b = bval[u_base[u]] + u_bc[u]
                if b > 0.0:
                    fa = u_a[u]
                    a = u_aconst[u] if fa < 0 else _dot(fa, v, e_ptr, e_dim, e_coef)
                    v[u_out[u]] += u_c[u] * a * b
        elif kind == 1:  # running sums
            for k in range(a0, a1):
                sums[k] += _dot(s_form[k], v, e_ptr, e_dim, e_coef)
                v[s_out[k]] = sums[k]
        else:  # one hardmax head: insert this position's line, then query
            h = a0
            m = np.int64(_dot(h_kx[h], v, e_ptr, e_dim, e_coef))
            bb = np.int64(_dot(h_ky[h], v, e_ptr, e_dim, e_coef))
            x = np.int64(_dot(h_q[h], v, e_ptr, e_dim, e_coef))
            nv = h_vptr[h + 1] - h_vptr[h]
            for k in range(nv):
                tmp[k] = _dot(h_vform[h_vptr[h] + k], v, e_ptr, e_dim, e_coef)
            base = h_base[h]
            vb = h_vbase[h]
            lo, hi = h_lo[h], h_hi[h]
            # --- Li Chao insert (max), children created on demand
            if n_count[h] == 0:
                n_m[base], n_b[base], n_l[base], n_r[base] = m, bb, -1, -1
                for k in range(nv):
                    n_vals[vb + k] = tmp[k]
                n_count[h] = 1
            else:
                idx = 0
                while True:
                    g = base + idx
                    mid = (lo + hi) // 2
                    left = m * lo + bb > n_m[g] * lo + n_b[g]
                    center = m * mid + bb > n_m[g] * mid + n_b[g]
                    if center:
                        m, n_m[g] = n_m[g], m
                        bb, n_b[g] = n_b[g], bb
                        off = vb + idx * nv
                        for k in range(nv):
                            t = n_vals[off + k]
                            n_vals[off + k] = tmp[k]
                            tmp[k] = t
                    if lo == hi:
                        break
                    if left != center:
                        child = n_l[g]
                        hi = mid
                        if child < 0:
                            child = n_count[h]
                            n_count[h] += 1
                            n_l[g] = child
                            cg = base + child
                            n_m[cg], n_b[cg], n_l[cg], n_r[cg] = m, bb, -1, -1
                            for k in range(nv):
                                n_vals[vb + child * nv + k] = tmp[k]
                            break
                    else:
                        child = n_r[g]
                        lo = mid + 1
                        if child < 0:
                            child = n_count[h]
                            n_count[h] += 1
                            n_r[g] = child
                            cg = base + child
                            n_m[cg], n_b[cg], n_l[cg], n_r[cg] = m, bb, -1, -1
                            for k in range(nv):
                                n_vals[vb + child * nv + k] = tmp[k]
                            break
                    idx = child
            # --- query at x
            if x < h_lo[h] or x > h_hi[h]:
                return -1000000 - h
            lo, hi = h_lo[h], h_hi[h]
            idx = 0
            best = -1
            bestv = np.int64(0)
            while idx >= 0:
                g = base + idx
                sc = n_m[g] * x + n_b[g]
                if best < 0 or sc > bestv:
                    best, bestv = idx, sc
                if lo == hi:
                    break
                mid = (lo + hi) // 2
                if x <= mid:
                    idx = n_l[g]
                    hi = mid
                else:
                    idx = n_r[g]
                    lo = mid + 1
            off = vb + best * nv
            for k in range(nv):
                v[h_out[h_vptr[h] + k]] = n_vals[off + k]
    if check:
        for i in range(r_form.shape[0]):
            z = _dot(r_form[i], v, e_ptr, e_dim, e_coef)
            if z < r_lo[i] or z > r_hi[i] or z != np.floor(z):
                return -1 - i
    return _dot(out_form, v, e_ptr, e_dim, e_coef)


@njit(cache=True)
def _run(toks, t_ptr, t_dim, t_val, pos0, out, v, d_pos, d_pos2,
         e_ptr, e_dim, e_coef, sched,
         h_q, h_kx, h_ky, h_lo, h_hi, h_vptr, h_vform, h_out,
         n_m, n_b, n_l, n_r, n_vals, n_count, h_base, h_vbase, tmp,
         s_form, s_out, sums, u_a, u_aconst, u_base, u_bc, b_form, bval, u_c, u_out, out_form, r_form, r_lo, r_hi, check):
    """Teacher-forced batch: feed toks (indices into the t_* embedding table)."""
    for i in range(toks.shape[0]):
        k = toks[i]
        y = _step(v, t_dim[t_ptr[k]:t_ptr[k + 1]], t_val[t_ptr[k]:t_ptr[k + 1]], pos0 + i,
                  d_pos, d_pos2, e_ptr, e_dim, e_coef, sched,
                  h_q, h_kx, h_ky, h_lo, h_hi, h_vptr, h_vform, h_out,
                  n_m, n_b, n_l, n_r, n_vals, n_count, h_base, h_vbase, tmp,
                  s_form, s_out, sums, u_a, u_aconst, u_base, u_bc, b_form, bval, u_c, u_out, out_form, r_form, r_lo, r_hi, check)
        out[i] = y
        if y < 0:
            return i + 1
    return toks.shape[0]


@njit(cache=True)
def _query(x, lo, hi, base, vb, nv, n_m, n_b, n_l, n_r, n_vals, out):
    """Read a head's cache at query x without inserting anything (the attention read alone)."""
    idx, best = 0, -1
    bestv = np.int64(0)
    while idx >= 0:
        g = base + idx
        sc = n_m[g] * x + n_b[g]
        if best < 0 or sc > bestv:
            best, bestv = idx, sc
        if lo == hi:
            break
        mid = (lo + hi) // 2
        if x <= mid:
            idx, hi = n_l[g], mid
        else:
            idx, lo = n_r[g], mid + 1
    for k in range(nv):
        out[k] = n_vals[vb + best * nv + k]


class Engine:
    """Stateful decoder: feed tokens, get the circuit's prediction after each."""

    def __init__(self, compiled: Compiled, max_positions: int = 1 << 20, check: bool = False):
        cp = self.cp = compiled
        self.check = check
        n_heads = len(cp.h_q)
        caps = np.minimum(cp.h_cap0, max_positions + 2)
        nv = cp.h_vptr[1:] - cp.h_vptr[:-1]
        self.h_base = np.concatenate([[0], np.cumsum(caps)[:-1]]).astype(np.int64)
        self.h_vbase = np.concatenate([[0], np.cumsum(caps * nv)[:-1]]).astype(np.int64)
        total = int(caps.sum())
        self.n_m = np.zeros(total, np.int64)
        self.n_b = np.zeros(total, np.int64)
        self.n_l = np.full(total, -1, np.int64)
        self.n_r = np.full(total, -1, np.int64)
        # values are integers below 2^24 (positions < 2^20), so float32 holds them exactly
        self.n_vals = np.zeros(int((caps * nv).sum()), np.float32)
        self.n_count = np.zeros(n_heads, np.int64)
        self.caps = caps
        self.nv = nv
        self.tmp = np.zeros(max(1, int(nv.max())), np.float64)
        self.sums = np.zeros(len(cp.s_form), np.float64)
        self.v = np.zeros(cp.n_dims, np.float64)
        self.bval = np.zeros(max(1, len(cp.b_form)), np.float64)
        self.pos = 0
        self.max_positions = max_positions
        self._emb: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        """Empty the context (keeps the allocations and the embedding cache)."""
        self.n_count[:] = 0
        self.sums[:] = 0.0
        self.pos = 0

    def save(self):
        """Snapshot the context (the used part of every head's cache)."""
        parts = []
        for h in range(len(self.n_count)):
            b, n, vb, nv = self.h_base[h], self.n_count[h], self.h_vbase[h], self.nv[h]
            parts.append((self.n_m[b:b + n].copy(), self.n_b[b:b + n].copy(),
                          self.n_l[b:b + n].copy(), self.n_r[b:b + n].copy(),
                          self.n_vals[vb:vb + n * nv].copy()))
        return parts, self.n_count.copy(), self.sums.copy(), self.v.copy(), self.pos

    def restore(self, snap) -> None:
        parts, count, sums, v, pos = snap
        for h, (m, b_, l, r, vals) in enumerate(parts):
            b, n, vb = self.h_base[h], len(m), self.h_vbase[h]
            self.n_m[b:b + n], self.n_b[b:b + n] = m, b_
            self.n_l[b:b + n], self.n_r[b:b + n] = l, r
            self.n_vals[vb:vb + len(vals)] = vals
        self.n_count[:] = count
        self.sums[:] = sums
        self.v[:] = v
        self.pos = pos

    def peek(self, head: str, xs) -> np.ndarray:
        """The values head ``head`` would return for each query in xs, given the context so far."""
        cp = self.cp
        h = cp.head_names.index(head)
        nv = int(cp.h_vptr[h + 1] - cp.h_vptr[h])
        out = np.zeros((len(xs), nv), np.float32)
        for i, x in enumerate(xs):
            _query(np.int64(x), cp.h_lo[h], cp.h_hi[h], self.h_base[h], self.h_vbase[h], nv,
                   self.n_m, self.n_b, self.n_l, self.n_r, self.n_vals, out[i])
        return out

    def embed(self, t: int):
        e = self._emb.get(t)
        if e is None:
            dims, vals = [], []
            for d, fn in self.cp.token_fns:
                x = fn(t)
                if x:
                    dims.append(d)
                    vals.append(float(x))
            e = (np.asarray(dims, np.int64), np.asarray(vals, np.float64))
            self._emb[t] = e
        return e

    def step(self, t: int) -> int:
        if self.pos >= self.max_positions:
            raise RuntimeError("context full")
        cp = self.cp
        dims, vals = self.embed(t)
        y = _step(self.v, dims, vals, self.pos, cp.d_pos, cp.d_pos2,
                  cp.e_ptr, cp.e_dim, cp.e_coef, cp.sched,
                  cp.h_q, cp.h_kx, cp.h_ky, cp.h_lo, cp.h_hi, cp.h_vptr, cp.h_vform, cp.h_out,
                  self.n_m, self.n_b, self.n_l, self.n_r, self.n_vals, self.n_count,
                  self.h_base, self.h_vbase, self.tmp,
                  cp.s_form, cp.s_out, self.sums,
                  cp.u_a, cp.u_aconst, cp.u_base, cp.u_bc, cp.b_form, self.bval, cp.u_c, cp.u_out, cp.out_form,
                  cp.r_form, cp.r_lo, cp.r_hi, self.check)
        if y <= -1000000:
            raise ValueError(f"position {self.pos}: query of head "
                             f"{cp.head_names[int(-y - 1000000)]} out of range")
        if y < 0 and self.check:
            i = int(-y - 1)
            raise ValueError(f"position {self.pos}: {cp.ranges[i][3]} outside "
                             f"[{cp.ranges[i][1]}, {cp.ranges[i][2]}]")
        if np.any(self.n_count > self.caps):
            raise RuntimeError("attention cache overflow")
        self.pos += 1
        return int(round(y))

    def feed(self, toks: list[int]) -> np.ndarray:
        """Teacher-forced: feed many tokens in one compiled loop; returns every prediction."""
        uniq = sorted(set(toks))
        index = {t: i for i, t in enumerate(uniq)}
        ptr, dims, vals = [0], [], []
        for t in uniq:
            d, v = self.embed(t)
            dims.extend(d.tolist())
            vals.extend(v.tolist())
            ptr.append(len(dims))
        cp = self.cp
        out = np.zeros(len(toks), np.float64)
        if self.pos + len(toks) > self.max_positions:
            raise RuntimeError("context full")
        n = _run(np.asarray([index[t] for t in toks], np.int64), np.asarray(ptr, np.int64),
                 np.asarray(dims, np.int64), np.asarray(vals, np.float64), self.pos, out, self.v,
                 cp.d_pos, cp.d_pos2, cp.e_ptr, cp.e_dim, cp.e_coef, cp.sched,
                 cp.h_q, cp.h_kx, cp.h_ky, cp.h_lo, cp.h_hi, cp.h_vptr, cp.h_vform, cp.h_out,
                 self.n_m, self.n_b, self.n_l, self.n_r, self.n_vals, self.n_count,
                 self.h_base, self.h_vbase, self.tmp, cp.s_form, cp.s_out, self.sums,
                 cp.u_a, cp.u_aconst, cp.u_base, cp.u_bc, cp.b_form, self.bval, cp.u_c, cp.u_out, cp.out_form, cp.r_form, cp.r_lo, cp.r_hi,
                 self.check)
        self.pos += n
        if n < len(toks) or (n and out[n - 1] < 0):
            raise ValueError(f"position {self.pos - 1}: range or query check failed ({out[n - 1]})")
        return np.rint(out).astype(np.int64)

    def dim(self, name: str) -> float:
        """Debug: the current value of a named dim (after the latest step)."""
        names = getattr(self.cp, "_names", None)
        if names is None:
            names = self.cp._names = {dm.name: d for d, dm in enumerate(self.cp.g.dims)}
        return self.v[names[name]]
