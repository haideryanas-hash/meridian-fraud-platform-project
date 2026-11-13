"""Synthetic payment-event simulator with injected fraud campaigns (BRD section 8, build-phase proxy).

Fraud typologies (all with ground truth):
  cnp           stolen-card carding bursts from a small attacker device/IP pool
  ato           account takeover -> A2A transfers to reused external cash-out accounts
  ring_in       victim -> mule payments (half ATO on a ring device, half authorised-push-payment scams)
  ring_cashout  mule -> onward transfer on a ring device
  ring_card     carding on ring devices
Legitimate noise: shared family devices/IPs, public-wifi hub IPs, travel, new devices, bursty customers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DAY

COUNTRY_P = np.array([.38, .14, .10, .08, .07, .07, .05, .04, .04, .03])
CENTROID = np.array([[20.6, 78.9], [39.8, -98.6], [54.0, -2.0], [51.2, 10.4], [1.35, 103.8],
                     [24.3, 54.3], [-10.8, -52.9], [9.1, 8.7], [61.5, 105.3], [14.1, 108.3]])
RISKY_CATS = np.array([16, 17, 18, 19])
_H = np.array([1] * 6 + [3, 3, 4] + [6] * 13 + [3, 3], float)
HOUR_P = _H / _H.sum()

_COLS = ["t", "a2a", "amount", "cust", "card", "dev", "ip", "mer", "ben", "ftype", "camp", "taint_dev", "taint_sender"]
_DEF = {"card": -1, "mer": -1, "ben": -1, "ftype": "", "camp": -1, "taint_dev": 0, "taint_sender": 0, "a2a": 0}


class _Rows:
    def __init__(self):
        self.d = {c: [] for c in _COLS}

    def add(self, **kw):
        for c in _COLS:
            self.d[c].append(kw.get(c, _DEF.get(c)))

    def frame(self):
        return pd.DataFrame(self.d)


def simulate(sim: dict, lab: dict) -> pd.DataFrame:
    rng = np.random.default_rng(sim["seed"])
    nC, nD, nM = sim["n_customers"], sim["n_days"], sim["n_merchants"]
    T = nD * DAY
    n_rings = sim["n_rings"]
    ring_N = rng.integers(14, 41, n_rings)
    ring_k = np.array([rng.integers(5, max(6, n // 2) + 1) for n in ring_N])
    nA = nC + int(ring_k.sum())            # accounts: customers then mules
    n_ext = 3000
    ext0 = nA                               # external beneficiary accounts [ext0, ext0+n_ext)

    # ---------------- account profiles
    home = rng.choice(10, nA, p=COUNTRY_P)
    signup_day = np.where(rng.random(nA) < 0.9, -rng.integers(0, 1500, nA), rng.integers(0, nD, nA)).astype(float)
    kyc = rng.choice([1, 2, 3], nA, p=[.2, .5, .3])
    tier = rng.choice([1, 2, 3], nA, p=[.7, .25, .05])
    region = rng.integers(0, 5, nA)
    anchor = CENTROID[home] + rng.normal(0, 1.0, (nA, 2))

    # ---------------- devices / IPs (legit)
    primary = np.arange(nC)
    secondary = nC + rng.integers(0, nC // 2, nC)
    ip_home = np.arange(nC)
    fam = np.where(rng.random(nC) < 0.08)[0]               # households sharing device + IP
    partner = rng.integers(0, nC, nC)
    primary[fam] = primary[partner[fam]]
    ip_home[fam] = ip_home[partner[fam]]
    home[fam] = home[partner[fam]]
    anchor[fam] = anchor[partner[fam]]
    D0 = nC + nC // 2
    att_dev = D0 + np.arange(3000)
    dev_ctr = D0 + 3000

    ip_country = [home[:nC].copy()]
    ip_rep = [rng.beta(1, 15, nC)]
    pool_n = 3000
    pool_ids = nC + np.arange(pool_n)
    pool_c = rng.choice(10, pool_n, p=COUNTRY_P)
    ip_country.append(pool_c); ip_rep.append(rng.beta(1, 12, pool_n))
    hub_n = 30
    hub_ids = nC + pool_n + np.arange(hub_n)
    ip_country.append(rng.choice(10, hub_n, p=COUNTRY_P)); ip_rep.append(rng.beta(2, 6, hub_n))
    att_n = 3000
    att_ip = nC + pool_n + hub_n + np.arange(att_n)
    ip_country.append(np.where(rng.random(att_n) < .6, rng.choice([6, 7, 8, 9], att_n), rng.choice(10, att_n, p=COUNTRY_P)))
    ip_rep.append(rng.beta(4, 5, att_n))
    ip_ctr = nC + pool_n + hub_n + att_n

    mcat = rng.integers(0, 20, nM)
    pop = 1.0 / np.arange(1, nM + 1) ** 0.8
    pop /= pop.sum()
    favs = rng.choice(nM, (nC, 6), p=pop)
    mu_c = rng.normal(3.4, 0.6, nC)
    card_risk = rng.beta(1.5, 15, nC)
    bl = np.where(rng.random((nC, 3)) < .5, rng.integers(0, nC, (nC, 3)), ext0 + rng.integers(0, n_ext, (nC, 3)))

    # ---------------- legitimate events (vectorised)
    rate = rng.lognormal(np.log(0.045), 0.8, nC)
    start_day = np.maximum(signup_day[:nC], 0.0)
    cnt = rng.poisson(rate * (nD - start_day))
    cust = np.repeat(np.arange(nC), cnt)
    n = cust.size
    d0 = start_day[cust]
    day = np.floor(d0 + rng.random(n) * (nD - d0)).astype(np.int64)
    t = day * DAY + rng.choice(24, n, p=HOUR_P) * 3600 + (rng.random(n) * 3600).astype(np.int64)
    a2a = (rng.random(n) < 0.25).astype(int)
    amount = np.round(np.clip(np.exp(mu_c[cust] + 1.2 * a2a + rng.normal(0, 0.7, n)), 1, None), 2)
    dev = np.where(rng.random(n) < .15, secondary[cust], primary[cust])
    r = rng.random(n)
    ip = np.empty(n, np.int64)
    m = r < .8
    ip[m] = ip_home[cust[m]]
    m_hub = r >= .97
    ip[m_hub] = rng.choice(hub_ids, m_hub.sum())
    m_pool = (r >= .8) & (r < .97)
    for c in range(10):
        sel = np.where(m_pool & (home[cust] == c))[0]
        same = pool_ids[pool_c == c]
        if same.size == 0:
            same = pool_ids
        pick = rng.choice(same, sel.size)
        anyp = rng.random(sel.size) < .03
        pick[anyp] = rng.choice(pool_ids, anyp.sum())
        ip[sel] = pick
    mer = np.where(rng.random(n) < .7, favs[cust, rng.integers(0, 6, n)], rng.choice(nM, n, p=pop))
    mer[a2a == 1] = -1
    card = np.where(a2a == 1, -1, cust)
    rnd = np.where(rng.random(n) < .5, rng.integers(0, nC, n), ext0 + rng.integers(0, n_ext, n))
    ben = np.where(rng.random(n) < .85, bl[cust, rng.integers(0, 3, n)], rnd)
    ben[a2a == 0] = -1
    big = rng.random(n) < 0.02
    amount = np.round(np.where(big, amount * rng.uniform(5, 20, n), amount), 2)       # legitimate large payments
    legit = pd.DataFrame({"t": t, "a2a": a2a, "amount": amount, "cust": cust, "card": card, "dev": dev, "ip": ip,
                          "mer": mer, "ben": ben, "ftype": "", "camp": -1, "taint_dev": 0, "taint_sender": 0})
    burst = legit.sample(frac=0.03, random_state=sim["seed"])                          # legitimate rapid bursts
    reps = []
    for j in range(1, 4):
        b = burst.sample(frac=1.0 / j, random_state=j).copy()
        b["t"] = b["t"] + rng.uniform(20, 600, len(b)).astype(np.int64)
        b["amount"] = np.round(b["amount"] * rng.uniform(.5, 1.5, len(b)), 2)
        reps.append(b)
    legit = pd.concat([legit] + reps, ignore_index=True)

    # ---------------- fraud campaigns (deliberately noisy: proxies, victim's own device, normal-range amounts)
    pool_by_c = {c: pool_ids[pool_c == c] for c in range(10)}

    def pick_ip(campaign_ips, victim, p_resi):
        if rng.random() < p_resi:                                  # residential proxy in the victim's own country
            return int(rng.choice(pool_by_c[int(home[victim])]))
        return int(rng.choice(campaign_ips))

    def pick_dev(campaign_devs, victim, p_own):
        return int(primary[victim]) if rng.random() < p_own else int(rng.choice(campaign_devs))

    rows = _Rows()
    camp = 0
    risky_m = np.where(np.isin(mcat, RISKY_CATS))[0]
    for _ in range(sim["n_cnp_campaigns"]):
        camp += 1
        dday = rng.uniform(1, nD - 1)
        devs = rng.choice(att_dev, rng.integers(1, 3), replace=False)
        ips = rng.choice(att_ip, rng.integers(1, 4), replace=False)
        elig = np.where(signup_day[:nC] < dday)[0]
        victims = rng.choice(elig, rng.integers(3, 9), replace=False)
        base = int(dday) * DAY
        for v in victims:
            hr = rng.integers(0, 6) if rng.random() < .35 else rng.choice(24, p=HOUR_P)
            t0 = base + hr * 3600 + rng.uniform(0, 3600)
            for j in range(rng.integers(1, 4)):
                mm = rng.choice(risky_m) if rng.random() < .35 else rng.integers(0, nM)
                own = rng.random() < .4
                rows.add(t=t0 + j * rng.uniform(30, 900), amount=np.exp(rng.normal(4.3, 0.9)), cust=int(v), card=int(v),
                         dev=int(primary[v]) if own else int(rng.choice(devs)), ip=pick_ip(ips, v, .5), mer=int(mm),
                         ftype="cnp", camp=camp, taint_dev=int(not own))
    ext_cash = ext0 + rng.choice(n_ext, 60, replace=False)
    for _ in range(sim["n_ato_campaigns"]):
        camp += 1
        dday = rng.uniform(1, nD - 1)
        devs = rng.choice(att_dev, rng.integers(1, 3), replace=False)
        ips = rng.choice(att_ip, rng.integers(1, 4), replace=False)
        elig = np.where(signup_day[:nC] < dday)[0]
        victims = rng.choice(elig, rng.integers(2, 7), replace=False)
        base = int(dday) * DAY
        for v in victims:
            t0 = base + rng.choice(24, p=HOUR_P) * 3600 + rng.uniform(0, 3600)
            own = rng.random() < .4                                    # session hijack / malware on the victim's device
            for j in range(rng.integers(1, 4)):
                amt = float(np.clip(np.exp(mu_c[v] + 1.2) * rng.uniform(1.5, 6), 80, 6000))
                rows.add(t=t0 + j * rng.uniform(60, 1200), a2a=1, amount=amt, cust=int(v),
                         dev=int(primary[v]) if own else int(rng.choice(devs)), ip=pick_ip(ips, v, .4),
                         ben=int(rng.choice(ext_cash)) if rng.random() < .5 else int(ext0 + rng.integers(0, n_ext)),
                         ftype="ato", camp=camp, taint_dev=int(not own))
    mule_ptr = nC
    for rg in range(n_rings):
        camp += 1
        N, k = int(ring_N[rg]), int(ring_k[rg])
        mules = np.arange(mule_ptr, mule_ptr + k)
        mule_ptr += k
        start_day = rng.uniform(5, max(6.0, nD - min(45.0, 0.3 * nD)))
        start, dur = start_day * DAY, rng.uniform(15, 35) * DAY
        rc = int(rng.integers(0, 10))
        signup_day[mules] = start_day - rng.uniform(0, 5, k)
        kyc[mules], tier[mules], home[mules] = 1, 2, rc
        anchor[mules] = CENTROID[rc] + rng.normal(0, 1, (k, 2))
        nd, ni = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        devs = dev_ctr + np.arange(nd); dev_ctr += nd
        mule_dev = dev_ctr + np.arange(k); dev_ctr += k                # some mules operate from their own device
        ips = ip_ctr + np.arange(ni); ip_ctr += ni
        ip_country.append(np.full(ni, rc)); ip_rep.append(rng.beta(3.5, 5.5, ni))
        elig = np.where(signup_day[:nC] < start_day)[0]
        n_in, n_card = int(round(.5 * N)), int(round(.2 * N))
        for ti in start + np.minimum(rng.exponential(dur / 3, n_in), dur):
            v, mi = int(rng.choice(elig)), int(rng.integers(0, k))
            mule = int(mules[mi])
            amt = float(np.exp(rng.normal(5.6, 0.7)))
            ato = rng.random() < .35                                    # most are authorised-push-payment scams
            rows.add(t=ti, a2a=1, amount=amt, cust=v, dev=int(rng.choice(devs)) if ato else int(primary[v]),
                     ip=int(rng.choice(ips)) if ato else int(ip_home[v]), ben=mule, ftype="ring_in", camp=camp,
                     taint_dev=int(ato))
            if rng.random() < .6:
                tgt = int(rng.choice(mules)) if rng.random() < .6 else int(ext0 + rng.integers(0, n_ext))
                if tgt == mule:
                    tgt = int(ext0 + rng.integers(0, n_ext))
                shared = rng.random() < .65
                rows.add(t=ti + rng.uniform(1, 12) * 3600, a2a=1, amount=amt * rng.uniform(.7, .98), cust=mule,
                         dev=int(rng.choice(devs)) if shared else int(mule_dev[mi]),
                         ip=int(rng.choice(ips)) if shared else int(rng.choice(pool_by_c[rc])),
                         ben=tgt, ftype="ring_cashout", camp=camp, taint_dev=int(shared), taint_sender=1)
        for ti in start + np.minimum(rng.exponential(dur / 3, n_card), dur):
            v = int(rng.choice(elig))
            own = rng.random() < .3
            rows.add(t=ti, amount=float(np.exp(rng.normal(4.3, 0.8))), cust=v, card=v,
                     dev=int(primary[v]) if own else int(rng.choice(devs)), ip=pick_ip(ips, v, .4),
                     mer=int(rng.choice(risky_m)) if rng.random() < .5 else int(rng.integers(0, nM)),
                     ftype="ring_card", camp=camp, taint_dev=int(not own))

    ev = pd.concat([legit, rows.frame()], ignore_index=True)
    ev = ev[ev["t"] < T].copy()
    ev["t"] = ev["t"].astype(np.int64)
    ev = ev.sort_values("t", kind="stable").reset_index(drop=True)

    ipc, ipr = np.concatenate(ip_country), np.concatenate(ip_rep)
    cu = ev["cust"].to_numpy()
    ev["ip_country"] = ipc[ev["ip"].to_numpy()]
    ev["ip_rep"] = np.clip(ipr[ev["ip"].to_numpy()] + rng.normal(0, .02, len(ev)), 0, 1)
    same = ev["ip_country"].to_numpy() == home[cu]
    nE = len(ev)
    ev["geo_lat"] = np.where(same, anchor[cu, 0] + rng.normal(0, .02, nE), CENTROID[ev["ip_country"].to_numpy(), 0] + rng.normal(0, 1, nE))
    ev["geo_lon"] = np.where(same, anchor[cu, 1] + rng.normal(0, .02, nE), CENTROID[ev["ip_country"].to_numpy(), 1] + rng.normal(0, 1, nE))
    cd = ev["card"].to_numpy()
    ev["bin_risk"] = np.where(cd >= 0, card_risk[np.clip(cd, 0, nC - 1)], 0.0)
    ev["merchant_cat"] = np.where(ev["mer"].to_numpy() >= 0, mcat[np.clip(ev["mer"].to_numpy(), 0, nM - 1)], -1)
    ev["signup_time"] = (signup_day[cu] * DAY).astype(np.int64)
    ev["kyc_level"], ev["risk_tier"], ev["home_country"], ev["region"] = kyc[cu], tier[cu], home[cu], region[cu]

    # ---------------- labels with delay (BRD 8: analyst 2-5d, chargeback 30-90d, scaled)
    isf = (ev["ftype"] != "").to_numpy()
    analyst = rng.random(nE) < lab["analyst_share"]
    dly = np.where(analyst, rng.uniform(*lab["analyst_delay_days"], nE), rng.uniform(*lab["chargeback_delay_days"], nE))
    ev["label_time"] = (ev["t"] + np.where(isf, dly, lab["maturity_days"]) * DAY).astype(np.int64)
    ev["is_fraud"] = isf.astype(int)

    # ---------------- legacy rules (selection bias: declined payments never reveal their label)
    hour = (ev["t"] // 3600) % 24
    legacy = (((ev["a2a"] == 1) & (ev["amount"] > 2000)) |
              ((ev["ip_rep"] > .55) & (ev["amount"] > 250) & (ev["a2a"] == 0)) |
              ((~same) & (ev["amount"] > 400) & (hour < 6)))
    ev["legacy_declined"] = legacy.astype(int)
    ev["label_observed"] = ((~legacy) | (rng.random(nE) < sim["holdout_rate"])).astype(int)

    ev = ev.rename(columns={"t": "event_time", "a2a": "is_a2a", "cust": "customer_id", "card": "card_id",
                            "dev": "device_id", "ip": "ip_id", "mer": "merchant_id", "ben": "beneficiary_id",
                            "ftype": "fraud_type", "camp": "campaign_id"})
    ev.insert(0, "txn_id", np.arange(nE))
    ev["amount"] = ev["amount"].round(2)
    return ev
