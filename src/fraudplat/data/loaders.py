"""Loaders for the three public proxy datasets (BRD section 8 / Appendix A). The raw files must be downloaded
from Kaggle by the user (see docs/DATASETS.md); nothing here downloads data.

BRD rules enforced here:
  * IEEE-CIS : time-ordered; TransactionID / raw TransactionDT are never model features; encoders fit on train only.
  * Elliptic : temporal split by time step (train 1-29, validation 30-34, test 35-49); unlabeled nodes are never targets.
  * PaySim   : the four balance columns are DROPPED (fraudulent transfers are cancelled, so balances leak the label).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _find(dirpath: Path, name: str) -> Path:
    hits = [dirpath / name] + list(dirpath.rglob(name))
    for h in hits:
        if h.exists():
            return h
    raise FileNotFoundError(f"{name} not found under {dirpath}. Download it from Kaggle first (docs/DATASETS.md).")


# ------------------------------------------------------------------ IEEE-CIS
def load_ieee_cis(path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    tr = pd.read_csv(_find(path, "train_transaction.csv"), nrows=nrows)
    try:
        ident = pd.read_csv(_find(path, "train_identity.csv"))
        tr = tr.merge(ident, on="TransactionID", how="left")
    except FileNotFoundError:
        pass
    return tr.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def time_split(t: np.ndarray, train: float = 0.6, val: float = 0.2):
    """Out-of-time masks by quantiles of time (no shuffling)."""
    a, b = np.quantile(t, [train, train + val])
    return t < a, (t >= a) & (t < b), t >= b


def ieee_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dt = df["TransactionDT"]
    out["amt_log"] = np.log1p(df["TransactionAmt"]).astype("float32")
    out["hour"] = ((dt // 3600) % 24).astype("float32")
    out["dow"] = ((dt // 86400) % 7).astype("float32")
    skip = {"isFraud", "TransactionID", "TransactionDT", "TransactionAmt"}
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c].astype("float32")
        else:                                              # frequency encoding learned on the training period only
            freq = df.loc[train_mask, c].value_counts(normalize=True)
            out[c + "_freq"] = df[c].map(freq).fillna(0).astype("float32")
    if {"card1", "addr1"} <= set(df.columns):
        uid = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        out["uid_freq"] = uid.map(uid[train_mask].value_counts(normalize=True)).fillna(0).astype("float32")
    return out


# ------------------------------------------------------------------ Elliptic
def load_elliptic(path) -> dict:
    path = Path(path)
    f = pd.read_csv(_find(path, "elliptic_txs_features.csv"), header=None)
    cls = pd.read_csv(_find(path, "elliptic_txs_classes.csv"))
    ed = pd.read_csv(_find(path, "elliptic_txs_edgelist.csv"))
    ids = f[0].to_numpy()
    cmap = {"1": 1, "2": 0, "unknown": -1}                 # Elliptic: class 1 = illicit, 2 = licit
    lab = cls.set_index(cls.columns[0])[cls.columns[1]].astype(str).map(cmap).reindex(ids).fillna(-1).astype(int).to_numpy()
    idx = pd.Series(np.arange(len(ids)), index=ids)
    ed = ed[ed.iloc[:, 0].isin(idx.index) & ed.iloc[:, 1].isin(idx.index)]
    return {"ids": ids, "time_step": f[1].to_numpy(), "X": f.iloc[:, 2:].to_numpy(np.float32), "y": lab,
            "src": idx[ed.iloc[:, 0]].to_numpy(), "dst": idx[ed.iloc[:, 1]].to_numpy()}


# ------------------------------------------------------------------ PaySim
BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def load_paysim(path, nrows: int | None = None, drop_balances: bool = True) -> pd.DataFrame:
    path = Path(path)
    csv = path if path.is_file() else next(iter(sorted(path.rglob("*.csv"))), None)
    if csv is None:
        raise FileNotFoundError(f"no CSV under {path}. Download PaySim from Kaggle first (docs/DATASETS.md).")
    df = pd.read_csv(csv, nrows=nrows)
    if drop_balances:
        df = df.drop(columns=[c for c in BALANCE_COLS if c in df.columns])
    return df.sort_values("step", kind="stable").reset_index(drop=True)


def paysim_to_events(df: pd.DataFrame, seed: int = 0, label_delay_days: float = 2.0) -> pd.DataFrame:
    """Map PaySim rows to the platform's canonical event schema so the streaming engine can replay them.
    PaySim has no device / IP: each account gets a private pseudo-device and pseudo-IP (neutral, no leakage)."""
    assert not (set(BALANCE_COLS) & set(df.columns)), "balance columns leak the label and must be dropped"
    rng = np.random.default_rng(seed)
    codes_o, uo = pd.factorize(df["nameOrig"])
    allnames = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
    mapping = pd.Series(np.arange(len(allnames)), index=allnames)
    cust = mapping[df["nameOrig"]].to_numpy()
    ben = mapping[df["nameDest"]].to_numpy()
    n = len(df)
    t = df["step"].to_numpy().astype(np.int64) * 3600 + rng.integers(0, 3600, n)
    order = np.argsort(t, kind="stable")
    is_a2a = df["type"].isin(["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]).to_numpy().astype(int)
    ev = pd.DataFrame({"event_time": t, "is_a2a": is_a2a, "amount": df["amount"].to_numpy(float), "customer_id": cust,
                       "card_id": np.where(is_a2a == 0, cust, -1), "device_id": cust, "ip_id": cust,
                       "merchant_id": np.where(is_a2a == 0, ben, -1), "beneficiary_id": np.where(is_a2a == 1, ben, -1),
                       "merchant_cat": df["type"].astype("category").cat.codes.to_numpy(), "ip_country": 0, "geo_lat": 0.0,
                       "geo_lon": 0.0, "ip_rep": 0.0, "bin_risk": 0.0, "signup_time": 0, "kyc_level": 2, "risk_tier": 1,
                       "home_country": 0, "is_fraud": df["isFraud"].to_numpy(int)})
    ev["label_time"] = ev["event_time"] + int(label_delay_days * 86400)
    ev["label_observed"], ev["taint_dev"], ev["taint_sender"] = 1, 0, ev["is_fraud"]
    ev = ev.iloc[order].reset_index(drop=True)
    ev.insert(0, "txn_id", np.arange(n))
    return ev
