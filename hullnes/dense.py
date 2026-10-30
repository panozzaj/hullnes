"""The circuit as ordinary dense transformer weights, run with ordinary softmax attention.

This is the check that nothing in engine.py is doing the computing: the same graph, lowered to
weight matrices, run as a textbook decoder with a plain KV cache.

    r_t = E[token_t] + W_pos [t, t^2] + e_one
    per layer:
        r += sum_h W_o,h  softmax_s(beta * (W_q,h r_t) . (W_k,h r_s)) W_v,h r_s      (s <= t)
        r += W_down (relu(W_gate r) * (W_up r))
    logits_v = 2 v (w_next . r) - v^2        (peaks at v = next)

Each head's query and key are 2-dimensional. Hardmax lookups become softmax at temperature
1 / beta: scores are integers that differ by at least 1, so the runner-up gets weight
<= e^-beta. A Cumsum head has zero query and key, so softmax averages the prefix, and one
ReGLU unit multiplies the mean by (t + 1). That mean is only approximately an integer in
floating point; how far the error travels is what ``verify`` measures.
"""

import numpy as np

from chipformer.dsl.graph import FFN, ONE, Cumsum, Expr, Graph, Lookup


def row(e: Expr, d: int) -> np.ndarray:
    r = np.zeros(d)
    for k, v in e.terms.items():
        r[k] = float(v)
    return r


class DenseModel:
    def __init__(self, g: Graph, beta: float = 100.0):
        self.g, self.beta = g, beta
        d0 = len(g.dims)
        n_scratch = sum(len(n.values) for n in g.nodes if isinstance(n, Cumsum))
        d = self.d = d0 + n_scratch
        self.n_layers = g.n_layers()
        names = {g.dims[k].name: k for k in g.pos_fns}
        self.w_pos = np.zeros((d, 2))
        self.w_pos[names["pos"], 0] = 1.0
        self.w_pos[names["pos^2"], 1] = 1.0
        self.layers = [{"heads": [], "units": []} for _ in range(self.n_layers)]
        scratch = d0
        for n in g.nodes:
            L = (n.level - 1) // 2
            if isinstance(n, Lookup):
                self.layers[L]["heads"].append({
                    "name": n.name,
                    "wq": np.stack([row(n.q, d), row(g.one, d)]),
                    "wk": np.stack([row(n.kx, d), row(n.ky, d)]),
                    "wv": np.stack([row(v, d) for v in n.values]),
                    "out": np.array(n.out)})
            elif isinstance(n, Cumsum):  # uniform head into scratch dims, then mean * (t + 1)
                outs = list(range(scratch, scratch + len(n.values)))
                scratch += len(n.values)
                self.layers[L]["heads"].append({
                    "name": n.name, "wq": np.zeros((2, d)), "wk": np.zeros((2, d)),
                    "wv": np.stack([row(v, d) for v in n.values]), "out": np.array(outs)})
                pos = Expr({names["pos"]: 1})
                for s_dim, o in zip(outs, n.out):
                    self.layers[L]["units"].append((1.0, Expr({s_dim: 1}), pos + 1, o))
            else:
                for c, a, b in n.terms:
                    self.layers[L]["units"].append((float(c), a, b, n.out))
        for lay in self.layers:
            u = lay.pop("units")
            lay["w_gate"] = np.stack([row(b, d) for _, _, b, _ in u]) if u else np.zeros((0, d))
            lay["w_up"] = np.stack([row(a, d) for _, a, _, _ in u]) if u else np.zeros((0, d))
            lay["coef"] = np.array([c for c, _, _, _ in u])
            lay["down"] = np.array([o for _, _, _, o in u], dtype=np.int64)
        self.w_next = row(g.outputs["next"], d)
        self.token_fns = list(g.token_fns.items())
        self._emb: dict[int, np.ndarray] = {}
        self.reset()

    def n_params(self) -> dict:
        nz = sum(int(np.count_nonzero(h[k])) for lay in self.layers for h in lay["heads"]
                 for k in ("wq", "wk", "wv"))
        nz += sum(int(np.count_nonzero(lay[k])) for lay in self.layers for k in ("w_gate", "w_up"))
        return {"d_model": self.d, "layers": self.n_layers, "nonzero": nz}

    def reset(self, capacity: int = 1 << 16):
        self.k_cache = [[np.zeros((capacity, 2)) for _ in lay["heads"]] for lay in self.layers]
        self.v_cache = [[np.zeros((capacity, len(h["wv"]))) for h in lay["heads"]]
                        for lay in self.layers]
        self.pos = 0

    def embed(self, t: int) -> np.ndarray:
        e = self._emb.get(t)
        if e is None:
            e = np.zeros(self.d)
            e[ONE] = 1.0
            for k, fn in self.token_fns:
                e[k] = float(fn(t))
            self._emb[t] = e
        return e

    def forward(self, toks: list[int]) -> np.ndarray:
        """Feed a chunk of tokens (appending to the KV cache); returns its residual rows."""
        n = len(toks)
        t = np.arange(self.pos, self.pos + n, dtype=np.float64)
        r = np.stack([self.embed(x) for x in toks]) + np.stack([t, t * t], 1) @ self.w_pos.T
        for L, lay in enumerate(self.layers):
            attn = np.zeros_like(r)
            for h, head in enumerate(lay["heads"]):
                q = r @ head["wq"].T
                end = self.pos + n
                self.k_cache[L][h][self.pos:end] = r @ head["wk"].T
                self.v_cache[L][h][self.pos:end] = r @ head["wv"].T
                k, v = self.k_cache[L][h][:end], self.v_cache[L][h][:end]
                s = q @ k.T  # (n, pos + n)
                s[np.arange(n)[:, None] < np.arange(-self.pos, n)[None, :]] = -np.inf  # causal
                s -= s.max(axis=1, keepdims=True)
                p = np.exp(self.beta * s)
                p /= p.sum(axis=1, keepdims=True)
                attn[:, head["out"]] += p @ v
            r = r + attn
            if len(lay["coef"]):
                hid = np.maximum(r @ lay["w_gate"].T, 0.0) * (r @ lay["w_up"].T) * lay["coef"]
                np.add.at(r.T, lay["down"], hid.T)
        self.pos += n
        return r

    def predict(self, r: np.ndarray, vocab: int) -> np.ndarray:
        """argmax over the whole vocabulary of logits 2 v y - v^2."""
        y = r @ self.w_next
        v = np.arange(vocab, dtype=np.float64)
        return np.array([int(np.argmax(2 * v * yi - v * v)) for yi in y])


def verify(rom, n_tokens: int, pc: int | None = None, buttons=None, log=print) -> tuple[int, int]:
    """Teacher-forced check of the dense softmax model against the reference trace.
    Returns (positions checked, mismatches)."""
    import time

    from hullnes import tokens as T
    from hullnes.circuit import Config, build
    from hullnes.run import boot, injected, run_reference

    trace, _, _ = run_reference(rom, frames=10**9, max_tokens=n_tokens, pc=pc, buttons=buttons)
    cfg = Config.for_board(rom.board)
    m = DenseModel(build(cfg))
    prompt = cfg.layout.prompt(rom.prg, T.cpu_state(boot(rom, pc)))[:-1]
    m.reset(len(prompt) + len(trace) + 8)
    log(f"dense model: {m.n_params()}; prompt {len(prompt):,} tokens, trace {len(trace):,}")
    t0 = time.time()
    for i in range(0, len(prompt), 512):
        m.forward(prompt[i:i + 512])
    log(f"prefill: {time.time() - t0:.0f}s")
    bad = checked = 0
    worst = 0.0
    for i, t in enumerate(trace):
        r = m.forward([t])
        y = float(r[0] @ m.w_next)
        worst = max(worst, abs(y - round(y)))
        if i + 1 < len(trace) and not injected(trace[i + 1]):
            checked += 1
            p = int(m.predict(r, T.VOCAB)[0])
            if p != trace[i + 1]:
                bad += 1
                if bad <= 3:
                    log(f"mismatch after token {i}: reference {T.name(trace[i + 1])}, dense {p}")
    log(f"checked {checked:,}, mismatches {bad}, max |next - round(next)| = {worst:.1e}")
    return checked, bad
