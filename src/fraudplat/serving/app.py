"""FastAPI scoring service.   uvicorn --factory fraudplat.serving.app:create_app_from_env"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ..config import DAY, artifacts_dir, data_dir, load_settings
from ..features.build import replay
from ..features.engine import FeatureEngine
from ..models.bundle import ModelBundle
from .scorer import Scorer


class PaymentEvent(BaseModel):
    txn_id: int
    event_time: int
    is_a2a: int
    amount: float
    customer_id: int
    card_id: int = -1
    device_id: int
    ip_id: int
    merchant_id: int = -1
    beneficiary_id: int = -1
    merchant_cat: int = -1
    ip_country: int
    geo_lat: float
    geo_lon: float
    ip_rep: float
    bin_risk: float = 0.0
    signup_time: int
    kyc_level: int
    risk_tier: int
    home_country: int


class LabelFeedback(PaymentEvent):
    """Confirmed-fraud disposition from case management / chargebacks (delayed feedback)."""
    label_time: int
    taint_dev: int = 0
    taint_sender: int = 0


def build_scorer(bundle_dir=None, warm_until_day: int | None = None, audit_path=None) -> Scorer:
    S = load_settings()
    bundle = ModelBundle.load(bundle_dir or artifacts_dir() / "models" / "v1")
    engine = FeatureEngine(S["features"]["scan_cap"])
    if warm_until_day:
        ev = pd.read_parquet(data_dir() / "events.parquet")
        replay(ev[ev.event_time < warm_until_day * DAY].reset_index(drop=True), engine, verbose=False)
    return Scorer(bundle, engine, audit_path=audit_path)

def create_app(scorer: Scorer) -> FastAPI:
    app = FastAPI(title="Meridian Pay fraud scoring", version=scorer.b.version)

    from pathlib import Path
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def serve_dashboard():
        html_path = Path(__file__).resolve().parents[3] / "index.html"
        return FileResponse(html_path)

    @app.post("/v1/score")
    def score(ev: PaymentEvent):
        return scorer.score(ev.model_dump())

    @app.post("/v1/labels", status_code=202)
    def label(fb: LabelFeedback):
        scorer.engine.schedule_label(fb.model_dump())
        return {"accepted": True}

    @app.get("/healthz")
    def health():
        return {"status": "ok", "model_version": scorer.b.version, "policy_version": scorer.policy.version,
                "circuit_open": scorer.breaker.opened_at is not None, "tier_errors": scorer.errors,
                "degraded": bool(scorer.errors) or scorer.breaker.opened_at is not None}

    @app.get("/v1/metrics")
    def metrics():
        return scorer.latency_summary()

    return app


def create_app_from_env() -> FastAPI:
    return create_app(build_scorer(os.environ.get("FRAUDPLAT_BUNDLE"), int(os.environ.get("FRAUDPLAT_WARM_UNTIL_DAY", 0)) or None,
                                   os.environ.get("FRAUDPLAT_AUDIT_LOG", "audit.jsonl")))
