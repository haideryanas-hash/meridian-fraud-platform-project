"""Inductive heterogeneous GraphSAGE (FR-06) in pure NumPy/SciPy (hand-written backprop, no PyTorch needed).

* Nodes: accounts, devices, IPs.  Relations (6): device->acct, acct->device, ip->acct, acct->ip,
  sender->beneficiary (who paid me), beneficiary->sender (whom I paid).  Mean aggregation with fixed fan-out
  (most-recent neighbours) = neighbour sampling.  Weights are relation-specific.
* Task: mule-account detection (account is beneficiary of / sender in a confirmed fraud payment).
* Inductive: the same weights score any later snapshot, including accounts that did not exist at training time.
* Leakage control: node features and edges use only events strictly before the snapshot time; labels come only
  from dispositions with label_time <= training snapshot; 2-fold CROSS-FITTING means an account is never scored
  by a model that saw its own label (so the stacker is not fed in-sample scores).
* Snapshots are published weekly (like the scheduled embedding publish in FR-06); each payment uses the latest
  snapshot <= its event time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DAY = 86400
R = 6
F_DIM = 17
UNKNOWN_LOGIT = -6.0


# --------------------------------------------------------------------------- graph data
class GraphData:
    def __init__(self, ev: pd.DataFrame):
        self.t = ev["event_time"].to_numpy()
        self.cust, self.ben = ev["customer_id"].to_numpy(), ev["beneficiary_id"].to_numpy()
        self.dev, self.ip = ev["device_id"].to_numpy(), ev["ip_id"].to_numpy()
        self.amt, self.a2a = ev["amount"].to_numpy(), ev["is_a2a"].to_numpy()
        self.n_acct = int(max(self.cust.max(), self.ben.max())) + 1
        self.n_dev, self.n_ip = int(self.dev.max()) + 1, int(self.ip.max()) + 1
        self.N = self.n_acct + self.n_dev + self.n_ip
        first = ev.drop_duplicates("customer_id").set_index("customer_id")
        self.signup = np.zeros(self.n_acct); self.kyc = np.zeros(self.n_acct); self.tier = np.zeros(self.n_acct)
        self.signup[first.index] = first["signup_time"].to_numpy()
        self.kyc[first.index] = first["kyc_level"].to_numpy()
        self.tier[first.index] = first["risk_tier"].to_numpy()
        w = np.bincount(self.ip, weights=ev["ip_rep"].to_numpy(), minlength=self.n_ip)
        self.ip_rep = w / np.maximum(np.bincount(self.ip, minlength=self.n_ip), 1)


def _relation(dst, src, tt, N, fanout):
    key = dst.astype(np.int64) * N + src
    order = np.lexsort((tt, key))
    ks, ts = key[order], tt[order]
    last = np.r_[ks[1:] != ks[:-1], True]
    ku, tu = ks[last], ts[last]
    du, su = ku // N, ku % N
    deg = np.bincount(du, minlength=N)
    o = np.lexsort((-tu, du))
    do, so = du[o], su[o]
    starts = np.r_[0, np.flatnonzero(do[1:] != do[:-1]) + 1]
    rank = np.arange(len(do)) - np.repeat(starts, np.diff(np.r_[starts, len(do)]))
    keep = rank < fanout
    dk, sk = do[keep], so[keep]
    cnt = np.bincount(dk, minlength=N).astype(np.float32)
    A = sp.csr_matrix((1.0 / cnt[dk], (dk, sk)), shape=(N, N), dtype=np.float32)
    return A, deg, (du, su)


class Snapshot:
    pass


def build_snapshot(g: GraphData, s: int, fanout: int) -> Snapshot:
    k = int(np.searchsorted(g.t, s))                       # events strictly before s
    cust, dev, ip, ben, tt = g.cust[:k], g.dev[:k], g.ip[:k], g.ben[:k], g.t[:k]
    D, I = g.n_acct + dev, g.n_acct + g.n_dev + ip
    am = ben >= 0
    specs = [(cust, D), (D, cust), (cust, I), (I, cust), (ben[am], cust[am]), (cust[am], ben[am])]
    A, degs, pairs = [], [], []
    for i, (dst, src) in enumerate(specs):
        t_e = tt if i < 4 else tt[am]
        a, d, p = _relation(dst, src, t_e, g.N, fanout)
        A.append(a); degs.append(d); pairs.append(p)
    X = np.zeros((g.N, F_DIM), np.float32)
    X[: g.n_acct, 0], X[g.n_acct: g.n_acct + g.n_dev, 1], X[g.n_acct + g.n_dev:, 2] = 1, 1, 1
    for r in range(R):
        X[:, 3 + r] = np.log1p(degs[r]) / 5
    sent = np.bincount(cust, minlength=g.n_acct)
    denom = np.maximum(sent, 1)
    X[: g.n_acct, 9] = np.log1p(sent) / 5
    X[: g.n_acct, 10] = np.bincount(cust, weights=np.log1p(g.amt[:k]), minlength=g.n_acct) / denom / 8
    X[: g.n_acct, 11] = np.bincount(cust, weights=g.a2a[:k], minlength=g.n_acct) / denom
    known = (sent > 0).astype(np.float32)
    X[: g.n_acct, 12] = np.clip((s - g.signup) / DAY / 365, 0, 5) / 5 * known
    X[: g.n_acct, 13] = g.kyc / 3 * known
    X[: g.n_acct, 14] = g.tier / 3 * known
    X[: g.n_acct, 15] = known
    X[g.n_acct + g.n_dev:, 16] = g.ip_rep[:] * (np.bincount(ip, minlength=g.n_ip) > 0)
    snap = Snapshot()
    snap.X, snap.A, snap.degs, snap.pairs = X, A, degs, pairs
    snap.has_edge = np.zeros(g.n_acct, bool)
    for r in range(R):
        snap.has_edge |= degs[r][: g.n_acct] > 0
    snap.dev_pairs = pairs[1]    # (dst=device node, src=account)
    return snap


# --------------------------------------------------------------------------- model
def _relu(z):
    return np.maximum(z, 0)


class HeteroSAGE:
    def __init__(self, hidden=32, seed=0, f_dim: int = F_DIM, n_rel: int = R):
        rng = np.random.default_rng(seed)
        H = hidden
        self.R, self.F = n_rel, f_dim

        def xav(*shape):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / (shape[-2] + shape[-1]))).astype(np.float32)

        self.p = {"Ws1": xav(f_dim, H), "Wr1": xav(n_rel, f_dim, H), "b1": np.zeros(H, np.float32),
                  "Ws2": xav(H, H), "Wr2": xav(n_rel, H, H), "b2": np.zeros(H, np.float32),
                  "w": xav(H, 1)[:, 0].copy(), "c": np.zeros((), np.float32)}

    def _layer1(self, X, M1):
        p = self.p
        Z1 = X @ p["Ws1"] + p["b1"]
        for r in range(self.R):
            Z1 += M1[r] @ p["Wr1"][r]
        return Z1

    def _layer2(self, H1, rows, A_rows):
        p = self.p
        M2 = [a @ H1 for a in A_rows]
        Z2 = H1[rows] @ p["Ws2"] + p["b2"]
        for r in range(self.R):
            Z2 += M2[r] @ p["Wr2"][r]
        return Z2, M2

    def predict_logits(self, snap: Snapshot, n_rows: int) -> np.ndarray:
        M1 = [a @ snap.X for a in snap.A]
        H1 = _relu(self._layer1(snap.X, M1))
        rows = np.arange(n_rows)
        A_rows = [a[:n_rows] for a in snap.A]
        Z2, _ = self._layer2(H1, rows, A_rows)
        return (_relu(Z2) @ self.p["w"] + self.p["c"]).astype(np.float32)

    def loss_grads(self, X, M1, idx, A_rows, AT_rows, yy, wt, wd=0.0):
        """Weighted BCE on rows `idx` and its analytic gradients (full backprop through both layers)."""
        p = self.p
        Z1 = self._layer1(X, M1)
        H1 = _relu(Z1)
        Z2, M2 = self._layer2(H1, idx, A_rows)
        H2 = _relu(Z2)
        s = H2 @ p["w"] + p["c"]
        prob = 1.0 / (1.0 + np.exp(-s))
        eps = 1e-7
        loss = float(-(wt * (yy * np.log(prob + eps) + (1 - yy) * np.log(1 - prob + eps))).sum())
        ds = ((prob - yy) * wt).astype(np.float32)
        g = {"w": H2.T @ ds, "c": ds.sum()}
        dZ2 = (ds[:, None] * p["w"][None, :]) * (Z2 > 0)
        g["Ws2"] = H1[idx].T @ dZ2 + wd * p["Ws2"]
        g["Wr2"] = np.stack([M2[r].T @ dZ2 for r in range(self.R)]) + wd * p["Wr2"]
        g["b2"] = dZ2.sum(0)
        dH1 = np.zeros_like(H1)
        dH1[idx] += dZ2 @ p["Ws2"].T
        for r in range(self.R):
            dH1 += AT_rows[r] @ (dZ2 @ p["Wr2"][r].T)
        dZ1 = dH1 * (Z1 > 0)
        g["Ws1"] = X.T @ dZ1 + wd * p["Ws1"]
        g["Wr1"] = np.stack([M1[r].T @ dZ1 for r in range(self.R)]) + wd * p["Wr1"]
        g["b1"] = dZ1.sum(0)
        return loss, g

    def fit(self, snap: Snapshot, y: np.ndarray, idx: np.ndarray, epochs=160, lr=0.01, wd=1e-4):
        p, X = self.p, snap.X
        M1 = [a @ X for a in snap.A]
        A_rows = [a[idx] for a in snap.A]
        AT_rows = [a.T.tocsr() for a in A_rows]
        yy = y[idx].astype(np.float32)
        pos = yy.sum()
        pw = float(np.clip(np.sqrt((len(yy) - pos) / max(pos, 1)), 1, 15))
        wt = np.where(yy == 1, pw, 1.0).astype(np.float32)
        wt /= wt.sum()
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v2 = {k: np.zeros_like(v) for k, v in p.items()}
        self.history = []
        for ep in range(1, epochs + 1):
            loss, g = self.loss_grads(X, M1, idx, A_rows, AT_rows, yy, wt, wd)
            self.history.append(loss)
            for k in p:                                    # Adam
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                mh, vh = m[k] / (1 - 0.9 ** ep), v2[k] / (1 - 0.999 ** ep)
                p[k] = (p[k] - lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return self


def fold_of(ids: np.ndarray) -> np.ndarray:
    return (((ids.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)) & np.uint64(1)).astype(int)


class CrossFitGNN:
    """Two models; an account is always scored by the model that did NOT train on its fold."""

    def __init__(self, models):
        self.models = models

    def score(self, g: GraphData, snap: Snapshot):
        lg = [m.predict_logits(snap, g.n_acct) for m in self.models]
        fold = fold_of(np.arange(g.n_acct))
        acct = np.where(fold == 0, lg[1], lg[0]).astype(np.float32)
        acct[~snap.has_edge] = UNKNOWN_LOGIT
        acct = np.clip(acct, -8, 8)
        dst_dev, src_acct = snap.dev_pairs                  # device node <- account
        dev_max = np.full(g.n_dev, UNKNOWN_LOGIT, np.float32)
        np.maximum.at(dev_max, dst_dev - g.n_acct, acct[src_acct])
        return acct, dev_max


def mule_accounts(ev: pd.DataFrame, s: int | None = None, observed_only: bool = True) -> np.ndarray:
    """Accounts linked to confirmed fraud as beneficiary / mule sender (labels visible by time s if given)."""
    m = ev["is_fraud"] == 1
    if observed_only:
        m &= ev["label_observed"] == 1
    if s is not None:
        m &= ev["label_time"] <= s
    sub = ev[m]
    ids = np.concatenate([sub.loc[sub["beneficiary_id"] >= 0, "beneficiary_id"].to_numpy(),
                          sub.loc[sub["taint_sender"] == 1, "customer_id"].to_numpy()])
    return np.unique(ids)


# --------------------------------------------------------------------------- snapshot store
class GNNStore:
    def __init__(self, s0, period, acct, dev_max):
        self.s0, self.period, self.acct, self.dev_max = s0, period, acct, dev_max

    def _k(self, t):
        return (np.clip((np.asarray(t) - self.s0) // self.period, -1, len(self.acct) - 1)).astype(int)

    def lookup(self, t, acct_id, ben_id, dev_id):
        """Vectorised. Returns (gnn_send, gnn_ben, gnn_dev) logits; UNKNOWN_LOGIT before the first snapshot."""
        t, acct_id, ben_id, dev_id = map(np.atleast_1d, (t, acct_id, ben_id, dev_id))
        k = self._k(t)
        ok = k >= 0
        kk = np.where(ok, k, 0)
        na, nd = self.acct.shape[1], self.dev_max.shape[1]
        send = np.where(ok & (acct_id < na), self.acct[kk, np.clip(acct_id, 0, na - 1)], UNKNOWN_LOGIT)
        ben = np.where(ok & (ben_id >= 0) & (ben_id < na), self.acct[kk, np.clip(ben_id, 0, na - 1)], UNKNOWN_LOGIT)
        dev = np.where(ok & (dev_id < nd), self.dev_max[kk, np.clip(dev_id, 0, nd - 1)], UNKNOWN_LOGIT)
        return send.astype(np.float32), ben.astype(np.float32), dev.astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, s0=self.s0, period=self.period, acct=self.acct, dev_max=self.dev_max)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(int(z["s0"]), int(z["period"]), z["acct"], z["dev_max"])


def build_gnn_store(ev: pd.DataFrame, cfg: dict, s_train: int, n_days: int, verbose=True):
    """Train cross-fitted GNN at s_train and publish weekly snapshot scores. Returns (store, report)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    g = GraphData(ev)
    snap0 = build_snapshot(g, s_train, cfg["fanout"])
    pos = mule_accounts(ev, s_train)
    y = np.zeros(g.N, np.float32)
    y[pos[pos < g.n_acct]] = 1
    nodes = np.flatnonzero(snap0.has_edge)
    fold = fold_of(nodes)
    models = []
    for k in (0, 1):
        tr = nodes[fold == k]
        if verbose:
            print(f"  GNN fold {k}: {len(tr)} train accounts, {int(y[tr].sum())} labelled mules", flush=True)
        models.append(HeteroSAGE(cfg["hidden"], cfg["seed"] + k).fit(snap0, y, tr, cfg["epochs"], cfg["lr"], cfg["weight_decay"]))
    cf = CrossFitGNN(models)

    period = cfg["snapshot_days"] * DAY
    times = list(range(s_train, n_days * DAY, period))
    accts, devs = [], []
    for s in times:
        snap = snap0 if s == s_train else build_snapshot(g, s, cfg["fanout"])
        a, d = cf.score(g, snap)
        accts.append(a); devs.append(d)
    store = GNNStore(s_train, period, np.stack(accts), np.stack(devs))

    # --- inductive evaluation: accounts that did not exist at s_train, scored at the last snapshot
    s_last = times[-1]
    snap_l = build_snapshot(g, s_last, cfg["fanout"])
    new = np.flatnonzero(snap_l.has_edge & ~snap0.has_edge)
    truth = np.zeros(g.n_acct, int)
    all_mules = mule_accounts(ev, observed_only=False)
    truth[all_mules[all_mules < g.n_acct]] = 1
    ap_gnn = float(average_precision_score(truth[new], store.acct[-1][new])) if truth[new].sum() else float("nan")
    # graph-statistics baseline: logistic regression on the same node features, no message passing, same cross-fit
    Xs0, Xsl = snap0.X[: g.n_acct], snap_l.X[: g.n_acct]
    base_scores = np.zeros(len(new))
    fnew = fold_of(new)
    for k in (0, 1):
        tr = nodes[fold == k]
        lr = LogisticRegression(class_weight="balanced", max_iter=500).fit(Xs0[tr], y[tr])
        base_scores += lr.decision_function(Xsl[new]) / 2
    ap_base = float(average_precision_score(truth[new], base_scores)) if truth[new].sum() else float("nan")
    report = {"snapshots": len(times), "train_positive_accounts": int(len(pos)), "new_accounts_evaluated": int(len(new)),
              "new_account_mules": int(truth[new].sum()), "pr_auc_gnn_inductive": ap_gnn,
              "pr_auc_graph_stats_baseline": ap_base,
              "base_rate": float(truth[new].mean()) if len(new) else float("nan")}
    return store, report
