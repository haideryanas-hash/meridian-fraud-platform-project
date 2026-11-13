# Meridian Pay - Real-Time Graph-Enhanced Payment Fraud Detection (BRD-DS-001)
export PYTHONPATH := src
PY := python3

.PHONY: install test data features train evaluate rings benchmark investigator report all serve clean

install:
	pip install -r requirements.txt

test:
	$(PY) -m pytest -q

data:                 ## simulate payments + inject fraud campaigns and mule rings
	$(PY) -c "from fraudplat.pipelines.data_pipeline import generate; generate()"

features:             ## point-in-time features by replaying events through the streaming engine
	$(PY) -c "from fraudplat.pipelines.data_pipeline import build_features; build_features()"

train:                ## GBDT + GNN + anomaly + stacker + calibration + policy shadow prices
	$(PY) -m fraudplat.pipelines.train

evaluate:             ## out-of-time metrics, ablation, policy simulation, drift, fairness
	$(PY) -m fraudplat.pipelines.evaluate

rings:                ## Leiden + personalised PageRank backtest (about 12 minutes on one core; stages run in separate processes to bound memory)
	$(PY) -m fraudplat.pipelines.rings --stage dev
	$(PY) -m fraudplat.pipelines.rings --stage test
	$(PY) -m fraudplat.pipelines.rings --stage test_labels
	$(PY) -m fraudplat.pipelines.rings --stage merge

benchmark:            ## live-path parity, latency, fallbacks, audit replay
	$(PY) -m fraudplat.pipelines.benchmark

investigator:         ## static HTML investigator view of detected rings
	$(PY) -m fraudplat.pipelines.investigator

report:               ## docs/RESULTS.md + model card from the measured JSON reports
	$(PY) -m fraudplat.pipelines.report

all: data features train evaluate rings benchmark investigator report

serve:                ## scoring API on :8000 (warm state from simulated history up to day 100)
	FRAUDPLAT_WARM_UNTIL_DAY=100 uvicorn --factory fraudplat.serving.app:create_app_from_env --port 8000

clean:
	rm -rf artifacts/* data/*.parquet
