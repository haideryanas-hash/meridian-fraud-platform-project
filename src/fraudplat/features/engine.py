"""Streaming feature engine (FR-02, FR-03, FR-04, FR-05).

One implementation serves both training (replay in event-time order) and live scoring, so feature
definitions are identical by construction. Every feature is computed from state built from events
strictly BEFORE the current one, and graph "taint" only reflects labels whose label_time <= now, which is
how point-in-time correctness and label delay are enforced. Kafka/Flink/Redis are replaced here by an
in-memory implementation with the same contract (process(event) -> feature vector).
"""
from __future__ import annotations

import heapq
import math
from bisect import bisect_left
from collections import defaultdict
from itertools import islice

import numpy as np

DAY = 86400
WINDOWS = [("1m", 60), ("10m", 600), ("1h", 3600), ("24h", 86400), ("7d", 7 * 86400), ("30d", 30 * 86400)]
ENTITIES = [("cust", "customer_id"), ("card", "card_id"), ("dev", "device_id"),
            ("ip", "ip_id"), ("ben", "beneficiary_id"), ("mer", "merchant_id")]

STATIC = ["amount", "log_amount", "hour", "dow", "is_a2a", "kyc_level", "risk_tier", "merchant_cat",
          "account_age_days", "ip_rep", "bin_risk", "ip_country_mismatch"]
VELOCITY = [f"{e}_{k}_{w}" for e, _ in ENTITIES for w, _ in WINDOWS for k in ("cnt", "sum")]
BEHAVIOUR = ["cust_prior_txns", "cust_amt_logmean", "cust_amt_logstd", "amount_z", "amount_over_max",
             "secs_since_last", "cust_burst_1h", "new_dev_for_cust", "new_mer_for_cust", "new_ben_for_cust",
             "n_cust_dev", "newdev_high_amt", "geo_dist_km", "geo_kmh"]
GRAPH = ["g_dev_taint", "g_ip_taint", "g_ben_taint", "g_acct_taint", "g_ben_dev_taint", "g_nbr_taint",
         "g_dev_n_accts", "g_ip_n_accts_7d", "g_ben_n_senders", "g_ben_n_senders_7d", "g_acct_n_devs",
         "g_acct_n_ips_7d", "g_nbr_accts_30d"]
FEATURE_NAMES = STATIC + VELOCITY + BEHAVIOUR + GRAPH
TABULAR_FEATURES = STATIC + VELOCITY + BEHAVIOUR           # no graph features (fallback / ablation)
N_FEATURES = len(FEATURE_NAMES)

_ZERO12 = [0.0] * 12


class FeatureEngine:
    def __init__(self, scan_cap: int = 400):
        self.scan_cap = scan_cap
        self.series = {name: {} for name, _ in ENTITIES}      # entity -> key -> [times, cumulative amounts]
        self.cstat: dict = {}                                  # (cust, channel) -> [n, sum_log, sumsq_log, max]
        self.clast: dict = {}                                  # cust -> (t, lat, lon)
        self.cdev, self.cmer, self.cben = defaultdict(set), defaultdict(set), defaultdict(set)
        self.dev_accts, self.ip_accts = defaultdict(dict), defaultdict(dict)
        self.acct_devs, self.acct_ips = defaultdict(set), defaultdict(dict)
        self.ben_senders = defaultdict(dict)
        self.t_dev, self.t_ip, self.t_acct = defaultdict(int), defaultdict(int), defaultdict(int)
        self._pending: list = []
        self._seq = 0
        self.last_time = -1

    # ------------------------------------------------------------------ labels (delayed feedback)
    def schedule_label(self, ev: dict):
        """Register a confirmed-fraud disposition; it only becomes visible once label_time has passed."""
        self._seq += 1
        heapq.heappush(self._pending, (ev["label_time"], self._seq, ev["device_id"], ev["ip_id"],
                                       ev["customer_id"], ev["beneficiary_id"], ev["taint_dev"], ev["taint_sender"]))

    def release(self, now: int):
        h = self._pending
        while h and h[0][0] <= now:
            _, _, dev, ip, sender, ben, td, ts = heapq.heappop(h)
            if td:
                self.t_dev[dev] += 1
                self.t_ip[ip] += 1
            if ts:
                self.t_acct[sender] += 1
            if ben >= 0:
                self.t_acct[ben] += 1

    # ------------------------------------------------------------------ features
    def _vel(self, name, key, t):
        s = self.series[name].get(key)
        if s is None:
            return _ZERO12
        times, cum = s
        n = len(times)
        tot = cum[n]
        out = []
        for _, w in WINDOWS:
            lo = bisect_left(times, t - w)
            out.append(float(n - lo))
            out.append(tot - cum[lo])
        return out

    def _count_recent(self, d, t, horizon):
        if not d:
            return 0.0
        if len(d) > self.scan_cap:
            return float(len(d))
        lim = t - horizon
        return float(sum(1 for lt in d.values() if lt >= lim))

    def compute(self, ev: dict) -> list:
        t, amt, c = ev["event_time"], ev["amount"], ev["customer_id"]
        dev, ip, ben, mer = ev["device_id"], ev["ip_id"], ev["beneficiary_id"], ev["merchant_id"]
        a2a = ev["is_a2a"]
        la = math.log1p(amt)
        out = [amt, la, float((t // 3600) % 24), float((t // DAY) % 7), float(a2a), float(ev["kyc_level"]),
               float(ev["risk_tier"]), float(ev["merchant_cat"]), min(max((t - ev["signup_time"]) / DAY, 0.0), 2000.0),
               ev["ip_rep"], ev["bin_risk"], float(ev["ip_country"] != ev["home_country"])]

        vc = None
        for name, col in ENTITIES:
            key = ev[col]
            v = self._vel(name, key, t) if key >= 0 else _ZERO12
            if name == "cust":
                vc = v
            out.extend(v)

        # --- behaviour vs the customer's own history
        cs = self.cstat.get((c, a2a))
        if cs and cs[0] >= 1:
            n0 = cs[0]
            mean = cs[1] / n0
            sd = math.sqrt(max(cs[2] / n0 - mean * mean, 0.0))
            z = (la - mean) / (sd + 0.3) if n0 >= 3 else 0.0
            over = amt / cs[3]
        else:
            n0, mean, sd, z, over = 0, la, 0.0, 0.0, 1.0
        last = self.clast.get(c)
        since = min(t - last[0], 30 * DAY) if last else 30 * DAY
        new_dev = 0.0 if dev in self.cdev.get(c, ()) else 1.0
        new_mer = 1.0 if (mer >= 0 and mer not in self.cmer.get(c, ())) else 0.0
        new_ben = 1.0 if (ben >= 0 and ben not in self.cben.get(c, ())) else 0.0
        if last:
            dy = (ev["geo_lat"] - last[1]) * 111.3
            dx = (ev["geo_lon"] - last[2]) * 111.3 * math.cos(math.radians((ev["geo_lat"] + last[1]) / 2))
            dist = min(math.hypot(dx, dy), 20000.0)
            kmh = min(dist / max((t - last[0]) / 3600.0, 1 / 60), 5000.0)
        else:
            dist = kmh = 0.0
        n_cd = len(self.cdev.get(c, ()))
        cs_list = self.series["cust"].get(c)
        out += [float(len(cs_list[0])) if cs_list else 0.0, mean, sd, z, over,
                float(since), vc[4] / (vc[8] / 168.0 + 0.1), new_dev, new_mer, new_ben, float(n_cd),
                new_dev * float(z > 2.0), dist, kmh]

        # --- graph features (time-stamped state; taint only from labels already released)
        td = self.t_dev.get(dev, 0)
        ti = self.t_ip.get(ip, 0)
        tb = self.t_acct.get(ben, 0) if ben >= 0 else 0
        ta = self.t_acct.get(c, 0)
        tbd = sum(1 for d in self.acct_devs.get(ben, ()) if self.t_dev.get(d, 0) > 0) if ben >= 0 else 0
        devs = set(islice(self.acct_devs.get(c, ()), 5))
        devs.add(dev)
        nbr, recent, seen = 0, 0, set()
        for d in devs:                                     # 2-hop neighbourhood via shared devices, with edge decay
            for a, lt in islice(self.dev_accts.get(d, {}).items(), 25):
                if a != c and a not in seen:
                    seen.add(a)
                    if self.t_acct.get(a, 0) > 0:
                        nbr += 1
                    if t - lt <= 30 * DAY:
                        recent += 1
        ip_accts = self.ip_accts.get(ip)
        out += [float(td), float(ti), float(tb), float(ta), float(tbd), float(nbr),
                float(len(self.dev_accts.get(dev, ()))), self._count_recent(ip_accts, t, 7 * DAY),
                float(len(self.ben_senders.get(ben, ()))) if ben >= 0 else 0.0,
                self._count_recent(self.ben_senders.get(ben), t, 7 * DAY) if ben >= 0 else 0.0,
                float(len(self.acct_devs.get(c, ()))), self._count_recent(self.acct_ips.get(c), t, 7 * DAY),
                float(recent)]
        return out

    def update(self, ev: dict):
        t, amt, c = ev["event_time"], ev["amount"], ev["customer_id"]
        dev, ip, ben, mer = ev["device_id"], ev["ip_id"], ev["beneficiary_id"], ev["merchant_id"]
        a2a = ev["is_a2a"]
        for name, col in ENTITIES:
            key = ev[col]
            if key >= 0:
                s = self.series[name].setdefault(key, [[], [0.0]])
                s[0].append(t)
                s[1].append(s[1][-1] + amt)
        la = math.log1p(amt)
        cs = self.cstat.get((c, a2a))
        if cs is None:
            self.cstat[(c, a2a)] = [1, la, la * la, amt]
        else:
            cs[0] += 1; cs[1] += la; cs[2] += la * la; cs[3] = max(cs[3], amt)
        self.clast[c] = (t, ev["geo_lat"], ev["geo_lon"])
        self.cdev[c].add(dev)
        if mer >= 0:
            self.cmer[c].add(mer)
        if ben >= 0:
            self.cben[c].add(ben)
            self.ben_senders[ben][c] = t
        self.acct_devs[c].add(dev)
        self.dev_accts[dev][c] = t
        self.ip_accts[ip][c] = t
        self.acct_ips[c][ip] = t
        self.last_time = t

    def process(self, ev: dict) -> list:
        """Decision-time path: release matured labels, compute features, then fold the event into state."""
        self.release(ev["event_time"])
        x = self.compute(ev)
        self.update(ev)
        return x
