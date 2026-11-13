"""Ring / coordinated-fraud detection (FR-10, BR-04): Leiden communities + personalised-PageRank risk propagation.

Graph (per detection run, trailing window): accounts, devices and IPs as nodes; edges account-device, account-IP
(IPs shared by more than `hub_cap` accounts are dropped as public-wifi hubs) and sender->beneficiary payments.
Seeds for risk propagation = infrastructure/accounts tied to confirmed fraud (label_time <= now) and, optionally,
payments the real-time model already flagged for review/decline. Communities are ranked by seed exposure,
shared-infrastructure density, exposed value and share of young accounts.
"""
from __future__ import annotations

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

DAY = 86400
OFF_D, OFF_I = 10_000_000, 20_000_000


def build_graph(win: pd.DataFrame, hub_cap: int):
    c = win["customer_id"].to_numpy()
    d = OFF_D + win["device_id"].to_numpy()
    i = OFF_I + win["ip_id"].to_numpy()
    b = win["beneficiary_id"].to_numpy()
    ipn = pd.DataFrame({"ip": i, "a": c}).drop_duplicates().groupby("ip").size()
    keep_ip = ~np.isin(i, ipn[ipn > hub_cap].index.to_numpy())
    am = b >= 0
    u = np.concatenate([c, c[keep_ip], c[am]])
    v = np.concatenate([d, i[keep_ip], b[am]])
    e = pd.DataFrame({"u": np.minimum(u, v), "v": np.maximum(u, v)}).groupby(["u", "v"]).size().reset_index(name="w")
    e = e[e.u != e.v]
    nodes = np.unique(np.concatenate([e.u.to_numpy(), e.v.to_numpy()]))
    ui, vi = np.searchsorted(nodes, e.u.to_numpy()), np.searchsorted(nodes, e.v.to_numpy())
    g = ig.Graph(n=len(nodes), edges=list(zip(ui.tolist(), vi.tolist())))
    g.es["weight"] = e.w.to_numpy(float).tolist()
    return g, nodes, (ui, vi, e.w.to_numpy())


def detect(win: pd.DataFrame, seed_acct: np.ndarray, seed_infra: np.ndarray, now: int, signup: np.ndarray,
           cfg: dict, return_edges: bool = False, min_score: float | None = None) -> list[dict]:
    """Return ranked community cases (score >= min_score, default cfg flag_score) for the window ending at `now`."""
    min_score = cfg["flag_score"] if min_score is None else min_score
    if len(win) < 10:
        return []
    g, nodes, (ui, vi, ew) = build_graph(win, cfg["hub_cap"])
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=0, n_iterations=2)
    comm = np.array(part.membership)
    is_acct = nodes < OFF_D
    seeds = np.flatnonzero(np.isin(nodes, np.concatenate([seed_acct, seed_infra]).astype(np.int64)))
    ppr = np.array(g.personalized_pagerank(reset_vertices=seeds.tolist(), damping=0.85, weights="weight")) \
        if len(seeds) else np.zeros(len(nodes))
    deg = np.array(g.degree())
    seed_flag = np.zeros(len(nodes), bool)
    seed_flag[seeds] = True
    sg = np.full(len(nodes), np.nan)
    acct_ids = nodes[is_acct]
    ok = acct_ids < len(signup)
    sg[np.flatnonzero(is_acct)[ok]] = signup[acct_ids[ok]]
    young = (now - sg <= 45 * DAY)

    # value flowing into each beneficiary account in the window
    am = win["beneficiary_id"].to_numpy() >= 0
    inflow = win[am].groupby("beneficiary_id")["amount"].sum()
    node_inflow = np.zeros(len(nodes))
    pos = np.searchsorted(nodes, inflow.index.to_numpy())
    pos_ok = (pos < len(nodes)) & (nodes[np.clip(pos, 0, len(nodes) - 1)] == inflow.index.to_numpy())
    node_inflow[pos[pos_ok]] = inflow.to_numpy()[pos_ok]
    senders = win[am].groupby("beneficiary_id")["customer_id"].nunique()      # mule-like: paid by several distinct senders
    node_multi = np.zeros(len(nodes), bool)
    pos2 = np.searchsorted(nodes, senders.index.to_numpy())
    ok2 = (pos2 < len(nodes)) & (nodes[np.clip(pos2, 0, len(nodes) - 1)] == senders.index.to_numpy())
    node_multi[pos2[ok2]] = senders.to_numpy()[ok2] >= 2

    df = pd.DataFrame({"comm": comm, "acct": is_acct, "infra": ~is_acct, "shared_infra": (~is_acct) & (deg >= 2),
                       "seed_acct": seed_flag & is_acct, "seed_infra": seed_flag & ~is_acct, "ppr": ppr,
                       "young": young & is_acct, "inflow": node_inflow, "multi": node_multi & is_acct})
    agg = df.groupby("comm").agg(n_accts=("acct", "sum"), n_infra=("infra", "sum"), shared=("shared_infra", "sum"),
                                 seed_acct=("seed_acct", "sum"), seed_infra=("seed_infra", "sum"), ppr_mass=("ppr", "sum"),
                                 young=("young", "sum"), exposed=("inflow", "sum"), multi=("multi", "sum")).reset_index()
    agg["density"] = agg["shared"] / agg["n_accts"].clip(lower=1)
    agg["young_share"] = agg["young"] / agg["n_accts"].clip(lower=1)
    agg["score"] = (1.5 * agg.seed_acct.clip(upper=3) + 1.0 * agg.seed_infra.clip(upper=2) + 1.5 * agg.density.clip(upper=2)
                    + 0.5 * np.log1p(agg.exposed / 1000) + 1.0 * agg.young_share + 1.0 * agg.multi.clip(upper=3)
                    + 3.0 * agg.ppr_mass)
    agg["flagged"] = (agg.n_accts >= cfg["min_accounts"]) & ((agg.seed_acct + agg.seed_infra) >= 1) & (agg.score >= min_score)
    out = []
    for r in agg[agg.flagged].sort_values("score", ascending=False).itertuples():
        m = comm == r.comm
        case = {"score": float(r.score), "n_accts": int(r.n_accts), "n_infra": int(r.n_infra), "shared_infra": int(r.shared),
                "seed_acct": int(r.seed_acct), "seed_infra": int(r.seed_infra), "mule_like": int(r.multi), "density": float(r.density),
                "exposed_value": float(r.exposed), "young_share": float(r.young_share), "ppr_mass": float(r.ppr_mass),
                "accounts": nodes[m & is_acct].tolist(), "infra": nodes[m & ~is_acct].tolist()}
        if return_edges:
            em = m[ui] & m[vi]
            case["edges"] = [(int(nodes[a]), int(nodes[b]), float(w)) for a, b, w in zip(ui[em], vi[em], ew[em])]
            case["ppr"] = {int(n): float(p) for n, p in zip(nodes[m], ppr[m])}
        out.append(case)
    return out
