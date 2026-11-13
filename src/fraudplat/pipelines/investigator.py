"""Investigator view (FR-11): ring cases with ego-graph, payment/score timeline, evidence and similar cases.
Static, self-contained HTML (no server needed)."""
from __future__ import annotations

import base64
import html
import io
import pickle

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

from ..config import DAY, artifacts_dir
from ..graph.rings import OFF_D, OFF_I
from .rings import ground_truth
from .common import load_events

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACT_COLOR = {0: "#9aa4b2", 1: "#e0a800", 2: "#e8590c", 3: "#c92a2a"}
ACT_NAME = {0: "approve", 1: "step-up", 2: "review", 3: "decline"}


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _ego_graph(case, max_nodes=55) -> str:
    G = nx.Graph()
    for u, v, w in case["edges"]:
        G.add_edge(u, v, weight=w)
    ppr = case["ppr"]
    if G.number_of_nodes() > max_nodes:
        keep = sorted(G.nodes, key=lambda n: -ppr.get(n, 0))[:max_nodes]
        G = G.subgraph(keep).copy()
    pos = nx.spring_layout(G, seed=3, k=1.3 / np.sqrt(max(G.number_of_nodes(), 1)), iterations=120)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=.35, width=[0.6 + 0.5 * np.log1p(d["weight"]) for *_, d in G.edges(data=True)], edge_color="#667085")
    shapes = [("o", lambda n: n < OFF_D, "#2f6fed", "account"), ("s", lambda n: OFF_D <= n < OFF_I, "#e8590c", "device"), ("^", lambda n: n >= OFF_I, "#2b8a3e", "IP")]
    mx = max(ppr.values()) if ppr else 1
    for shape, f, col, lab in shapes:
        nodes = [n for n in G.nodes if f(n)]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shape, node_color=col, ax=ax, label=lab,
                                   node_size=[60 + 380 * ppr.get(n, 0) / mx for n in nodes], edgecolors="white", linewidths=.8)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.axis("off")
    return _b64(fig)


def _timeline(case, dec, accts) -> str:
    d = case["detected_day"]
    lo, hi = (d - 16) * DAY, (d + 2) * DAY
    m = (dec.event_time >= lo) & (dec.event_time < hi) & (dec.customer_id.isin(accts) | dec.beneficiary_id.isin(accts))
    s = dec[m]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for a, g in s.groupby("action"):
        ax.scatter(g.event_time / DAY, g.p_cal, s=14 + 20 * np.log1p(g.amount) / 4, c=ACT_COLOR[a], label=ACT_NAME[a], alpha=.85, edgecolors="none")
    f = s[s.is_fraud == 1]
    ax.scatter(f.event_time / DAY, f.p_cal, s=70, facecolors="none", edgecolors="black", linewidths=.7, label="confirmed fraud (ground truth)")
    ax.axvline(d, color="#c92a2a", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("calibrated fraud probability")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    ax.grid(alpha=.2)
    return _b64(fig)


def build(out_path=None, top=6) -> str:
    cases = pickle.load(open(artifacts_dir() / "ring_cases.pkl", "rb"))
    dec = pd.read_parquet(artifacts_dir() / "decisions.parquet")
    rings, _ = ground_truth(load_events())
    ranked = sorted(cases.values(), key=lambda c: -c["score"])
    feat = np.array([[np.log1p(c["n_accts"]), c["density"], c["young_share"], np.log1p(c["exposed_value"]), c["mule_like"]] for c in ranked])
    z = (feat - feat.mean(0)) / (feat.std(0) + 1e-9) if len(ranked) > 1 else feat
    cards = []
    for i, c in enumerate(ranked[:top]):
        accts = np.array([a for a in c["accounts"]])
        ttd = (c["detected_day"] * DAY - rings[c["ring_id"]]["first_fraud"]) / DAY
        sims = np.argsort(np.linalg.norm(z - z[i], axis=1))[1:4] if len(ranked) > 1 else []
        top_acc = sorted(c["accounts"], key=lambda a: -c["ppr"].get(a, 0))[:6]
        rows = "".join(f"<tr><td>A{a}</td><td>{c['ppr'].get(a, 0):.4f}</td></tr>" for a in top_acc)
        sim = "".join(f"<li>ring R{ranked[j]['ring_id']} — score {ranked[j]['score']:.1f}, {ranked[j]['n_accts']} accounts</li>" for j in sims)
        cards.append(f"""<section class="case"><header><h2>Case {i + 1} <span class="tag">score {c['score']:.1f}</span></h2>
<p class="meta">Flagged on day {c['detected_day']} · simulation ground truth: ring R{c['ring_id']}, first fraud day {rings[c['ring_id']]['first_fraud'] / DAY:.1f}, time-to-detect {ttd:.1f} d</p></header>
<div class="kpis"><div><b>{c['n_accts']}</b><span>accounts</span></div><div><b>{c['shared_infra']}</b><span>shared devices/IPs</span></div>
<div><b>{c['mule_like']}</b><span>mule-like accounts</span></div><div><b>${c['exposed_value']:,.0f}</b><span>exposed value</span></div>
<div><b>{c['young_share']:.0%}</b><span>young accounts</span></div><div><b>{c['seed_acct'] + c['seed_infra']}</b><span>confirmed/flagged seeds</span></div></div>
<div class="figs"><figure><img alt="ego graph" src="data:image/png;base64,{_ego_graph(c)}"><figcaption>Ego-graph (node size = risk propagated by personalised PageRank)</figcaption></figure>
<figure><img alt="timeline" src="data:image/png;base64,{_timeline(c, dec, accts)}"><figcaption>Score timeline of payments touching these accounts</figcaption></figure></div>
<div class="cols"><div><h3>Highest-risk accounts</h3><table><tr><th>account</th><th>propagated risk</th></tr>{rows}</table></div>
<div><h3>Similar cases</h3><ul>{sim or '<li>none</li>'}</ul></div></div></section>""")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ring investigator view</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#14213d;--mut:#5b6573;--line:#e3e7ee;--acc:#2f6fed}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f141b;--card:#171e28;--ink:#e8edf5;--mut:#9aa6b6;--line:#26303d;--acc:#7aa2ff}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 18px 60px}} h1{{font-size:26px;margin:0 0 4px}}
.lead{{color:var(--mut);margin:0 0 22px}} .case{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:0 0 22px}}
h2{{margin:0;font-size:19px}} .tag{{font-size:12px;background:var(--acc);color:#fff;border-radius:99px;padding:2px 9px;vertical-align:middle;margin-left:6px}}
.meta{{color:var(--mut);margin:4px 0 14px;font-size:13px}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:14px}}
.kpis div{{border:1px solid var(--line);border-radius:10px;padding:8px 10px}} .kpis b{{display:block;font-size:20px}} .kpis span{{color:var(--mut);font-size:12px}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}} figure{{margin:0;background:#fff;border-radius:10px;padding:8px;border:1px solid var(--line)}}
figure img{{max-width:100%;height:auto}} figcaption{{color:#5b6573;font-size:12px;margin-top:4px}} .cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-top:14px}}
h3{{font-size:14px;margin:0 0 6px}} table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:13px}} ul{{margin:0;padding-left:18px}}
</style></head><body><main><h1>Ring investigator view</h1>
<p class="lead">Top {min(top, len(ranked))} of {len(ranked)} detected coordinated-fraud rings on the synthetic out-of-time test window, ranked by community risk score
(Leiden communities + personalised-PageRank risk propagation). Ground-truth fields are shown only because the data is simulated.</p>
{''.join(cards)}</main></body></html>"""
    out = out_path or artifacts_dir() / "reports" / "investigator.html"
    open(out, "w").write(doc)
    return str(out)


if __name__ == "__main__":
    print(build())
