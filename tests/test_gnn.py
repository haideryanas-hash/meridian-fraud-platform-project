import numpy as np

from fraudplat.config import DAY
from fraudplat.graph.gnn import (R, GraphData, HeteroSAGE, build_snapshot, fold_of, mule_accounts, GNNStore)


def _setup(ev):
    g = GraphData(ev)
    snap = build_snapshot(g, 30 * DAY, fanout=6)
    pos = mule_accounts(ev, 30 * DAY)
    y = np.zeros(g.N, np.float32)
    y[pos[pos < g.n_acct]] = 1
    nodes = np.flatnonzero(snap.has_edge)
    return g, snap, y, nodes


def test_backprop_matches_numerical_gradient(small_events):
    g, snap, y, nodes = _setup(small_events)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(nodes, 120, replace=False))
    yy = y[idx].copy(); yy[:5] = 1                       # make sure positives exist
    model = HeteroSAGE(8, seed=1)
    for k in model.p:
        model.p[k] = model.p[k].astype(np.float64)
    X = snap.X.astype(np.float64)
    M1 = [a.astype(np.float64) @ X for a in snap.A]
    A_rows = [a[idx].astype(np.float64) for a in snap.A]
    AT_rows = [a.T.tocsr() for a in A_rows]
    wt = np.where(yy == 1, 3.0, 1.0); wt /= wt.sum()
    _, g_an = model.loss_grads(X, M1, idx, A_rows, AT_rows, yy.astype(np.float64), wt)
    for name, pick in [("Ws2", (2, 3)), ("Wr2", (1, 2, 3)), ("Ws1", (4, 2)), ("Wr1", (4, 5, 1)), ("w", (3,)), ("b1", (2,))]:
        arr = model.p[name]
        old = arr[pick].copy()
        arr[pick] = old + 1e-5; lp, _ = model.loss_grads(X, M1, idx, A_rows, AT_rows, yy.astype(np.float64), wt)
        arr[pick] = old - 1e-5; lm, _ = model.loss_grads(X, M1, idx, A_rows, AT_rows, yy.astype(np.float64), wt)
        arr[pick] = old
        num = (lp - lm) / 2e-5
        assert np.isclose(g_an[name][pick], num, rtol=2e-3, atol=1e-6), (name, g_an[name][pick], num)


def test_training_reduces_loss(small_events):
    g, snap, y, nodes = _setup(small_events)
    m = HeteroSAGE(16, seed=0).fit(snap, y, nodes, epochs=40, lr=0.02)
    assert m.history[-1] < m.history[0]


def test_snapshot_uses_only_past_events(small_events):
    g = GraphData(small_events)
    s = 20 * DAY
    snap = build_snapshot(g, s, 6)
    k = int(np.searchsorted(g.t, s))
    seen = set(g.cust[:k]) | set(g.ben[:k][g.ben[:k] >= 0])
    future_only = set(g.cust[k:]) - seen
    ids = np.array(sorted(future_only))[:50]
    assert not snap.has_edge[ids].any()


def test_cross_fit_never_scores_with_own_fold():
    ids = np.arange(1000)
    f = fold_of(ids)
    assert 400 < f.sum() < 600                           # roughly balanced folds


def test_store_before_first_snapshot_is_unknown():
    store = GNNStore(100, 10, np.zeros((2, 5), np.float32), np.zeros((2, 4), np.float32))
    s, b, d = store.lookup([50], [1], [-1], [1])
    assert s[0] == -6.0 and b[0] == -6.0 and d[0] == -6.0
