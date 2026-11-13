"""Settings loader and shared constants."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DAY = 86400

# Columns the feature engine may read when scoring (what the payment switch sends at decision time).
EVENT_COLS = [
    "txn_id", "event_time", "is_a2a", "amount", "customer_id", "card_id", "device_id", "ip_id",
    "merchant_id", "beneficiary_id", "merchant_cat", "ip_country", "geo_lat", "geo_lon", "ip_rep",
    "bin_risk", "signup_time", "kyc_level", "risk_tier", "home_country",
]
# Information that only exists after the fact (labels / ground truth). Never read by the feature engine.
LABEL_COLS = [
    "is_fraud", "fraud_type", "campaign_id", "label_time", "label_observed", "legacy_declined",
    "taint_dev", "taint_sender",
]


def load_settings(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_yaml(rel: str) -> dict:
    with open(ROOT / rel) as f:
        return yaml.safe_load(f)


def config_hash(*objs) -> str:
    return hashlib.sha256(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:12]


def artifacts_dir() -> Path:
    p = ROOT / "artifacts"
    p.mkdir(exist_ok=True)
    return p


def data_dir() -> Path:
    p = ROOT / "data"
    p.mkdir(exist_ok=True)
    return p
