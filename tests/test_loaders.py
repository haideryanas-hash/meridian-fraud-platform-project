"""Loaders and public-dataset protocols are tested on tiny mock files in the real Kaggle formats."""
import numpy as np
import pandas as pd
import pytest

from fraudplat.data.loaders import (BALANCE_COLS, ieee_features, load_elliptic, load_ieee_cis, load_paysim,
                                    paysim_to_events, time_split)


def _ieee(tmp, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(86400, 86400 * 180, n))
    fraud = (rng.random(n) < 0.04).astype(int)
    tr = pd.DataFrame({"TransactionID": rng.permutation(n) + 1000, "isFraud": fraud, "TransactionDT": dt,
                       "TransactionAmt": np.exp(rng.normal(4 + fraud, 1, n)), "ProductCD": rng.choice(list("WCRSH"), n),
                       "card1": rng.integers(1000, 1200, n), "card4": rng.choice(["visa", "mastercard"], n),
                       "addr1": rng.integers(100, 130, n), "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
                       "C1": rng.poisson(1 + 3 * fraud), "D1": rng.normal(size=n), "V1": rng.normal(fraud, 1, n)})
    tr.to_csv(tmp / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": tr.TransactionID[::2], "DeviceType": rng.choice(["mobile", "desktop"], n // 2)}).to_csv(tmp / "train_identity.csv", index=False)


def test_ieee_loader_is_time_ordered_and_features_never_use_id_or_dt(tmp_path):
    _ieee(tmp_path)
    df = load_ieee_cis(tmp_path)
    assert df.TransactionDT.is_monotonic_increasing
    tr, va, te = time_split(df.TransactionDT.to_numpy())
    assert df.TransactionDT[tr].max() <= df.TransactionDT[va].min() and df.TransactionDT[va].max() <= df.TransactionDT[te].min()
    X = ieee_features(df, tr)
    assert not {"TransactionID", "TransactionDT", "isFraud"} & set(X.columns)
    # frequency encoders are learned on train only: a category that only appears in test gets 0
    df2 = df.copy(); df2.loc[te, "ProductCD"] = "ZZZ"
    assert (ieee_features(df2, tr).loc[te, "ProductCD_freq"] == 0).all()


def test_ieee_end_to_end(tmp_path, monkeypatch):
    from fraudplat.pipelines import public_datasets as pdx
    monkeypatch.setattr(pdx, "artifacts_dir", lambda: tmp_path)
    _ieee(tmp_path)
    res = pdx.run_ieee(tmp_path, n_rounds=60, verbose=False)
    assert res["test"]["pr_auc"] > res["prevalence"] and "baseline_logistic" in res


def _elliptic(tmp, per_step=60, steps=49, seed=0):
    rng = np.random.default_rng(seed)
    rows, cls, edges, nid = [], [], [], 1
    for s in range(1, steps + 1):
        ids = list(range(nid, nid + per_step)); nid += per_step
        illicit = rng.random(per_step) < 0.12
        for i, ill in zip(ids, illicit):
            rows.append([i, s] + list(rng.normal(ill * 1.5, 1, 6)))
            cls.append([i, "1" if ill else ("2" if rng.random() < .7 else "unknown")])
        for _ in range(per_step * 2):
            a, b = rng.choice(ids, 2, replace=False)
            edges.append([a, b])
    d = tmp / "elliptic_bitcoin_dataset"; d.mkdir()
    pd.DataFrame(rows).to_csv(d / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame(cls, columns=["txId", "class"]).to_csv(d / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(d / "elliptic_txs_edgelist.csv", index=False)


def test_elliptic_loader_and_protocol(tmp_path, monkeypatch):
    from fraudplat.pipelines import public_datasets as pdx
    monkeypatch.setattr(pdx, "artifacts_dir", lambda: tmp_path)
    _elliptic(tmp_path)
    d = load_elliptic(tmp_path)                                       # finds the nested folder like the Kaggle zip
    assert set(np.unique(d["y"])) == {-1, 0, 1} and d["X"].shape[1] == 6
    assert (d["time_step"][d["src"]] == d["time_step"][d["dst"]]).all()
    res = pdx.run_elliptic(tmp_path, epochs=40, verbose=False)
    for k in ("lightgbm_node_features_only", "lightgbm_plus_neighbour_means", "graphsage_inductive"):
        assert res[k]["pr_auc"] > res["illicit_share_labelled"]


def _paysim(tmp, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"], n)
    fraud = ((typ == "TRANSFER") | (typ == "CASH_OUT")) & (rng.random(n) < .05)
    df = pd.DataFrame({"step": np.sort(rng.integers(1, 700, n)), "type": typ, "amount": np.exp(rng.normal(5 + 2 * fraud, 1, n)),
                       "nameOrig": ["C%d" % i for i in rng.integers(0, 800, n)], "oldbalanceOrg": 1.0, "newbalanceOrig": 0.0,
                       "nameDest": ["C%d" % i for i in rng.integers(0, 800, n)], "oldbalanceDest": 1.0, "newbalanceDest": 2.0,
                       "isFraud": fraud.astype(int), "isFlaggedFraud": 0})
    df.to_csv(tmp / "PS_log.csv", index=False)


def test_paysim_drops_balance_columns_and_maps_to_events(tmp_path, monkeypatch):
    from fraudplat.pipelines import public_datasets as pdx
    monkeypatch.setattr(pdx, "artifacts_dir", lambda: tmp_path)
    _paysim(tmp_path)
    df = load_paysim(tmp_path)
    assert not set(BALANCE_COLS) & set(df.columns)
    ev = paysim_to_events(df)
    assert ev.event_time.is_monotonic_increasing and ev.txn_id.is_unique
    with pytest.raises(AssertionError):
        paysim_to_events(load_paysim(tmp_path, drop_balances=False))   # refuses leaky input
    res = pdx.run_paysim(tmp_path, max_rows=0, verbose=False)
    assert res["balance_columns_used"] is False and res["engine_events_per_second"] > 0
