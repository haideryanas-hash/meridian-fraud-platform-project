# Getting the public datasets

The BRD lists three public proxies (Appendix A). **No data is bundled with this project and none can be downloaded
automatically** (Kaggle requires your own account). The platform itself runs without them on the built-in simulator
(`make all`); these datasets are for the extra benchmarks in `fraudplat.pipelines.public_datasets`.

## 1. Kaggle API token (once)
1. Create a free account at kaggle.com.
2. Profile -> Settings -> API -> **Create New Token**. This downloads `kaggle.json`.
3. Put it in `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`) and on Linux/macOS run `chmod 600 ~/.kaggle/kaggle.json`.
4. `pip install kaggle`

> Use **your own** token. Some public notebooks download somebody else's `kaggle.json` from a shared link - do not do that.

## 2. Download
```bash
mkdir -p data/raw && cd data/raw
# IEEE-CIS: first open https://www.kaggle.com/c/ieee-fraud-detection and accept the competition rules, otherwise the download is refused
kaggle competitions download -c ieee-fraud-detection -p ieee_cis && unzip -o 'ieee_cis/*.zip' -d ieee_cis
kaggle datasets download -d ellipticco/elliptic-data-set -p elliptic --unzip
kaggle datasets download -d ealaxi/paysim1 -p paysim --unzip
```
Only the `train_*.csv` files of IEEE-CIS are needed (the test files carry no labels). Check each dataset's licence / competition rules before use.

## 3. Run (BRD protocol: temporal splits, no leakage, no SMOTE)
```bash
python -m fraudplat.pipelines.public_datasets ieee     --dir data/raw/ieee_cis
python -m fraudplat.pipelines.public_datasets elliptic --dir data/raw/elliptic
python -m fraudplat.pipelines.public_datasets paysim   --dir data/raw/paysim --max-rows 1000000
```
Results are written to `artifacts/reports/public_*.json`.

## What the protocol does differently from popular notebooks
| Dataset | Common notebook practice | Here |
|---|---|---|
| IEEE-CIS | random or ad-hoc split, `TransactionID` kept as a feature, scalers/PCA fit separately on test | time-ordered 60/20/20 split, IDs and raw time never features, encoders fit on the training period only |
| Elliptic | threshold tuned on the test set | train steps 1-29, **validation 30-34 picks the threshold**, test 35-49 |
| PaySim | balance-difference features, random split, SMOTE | balance columns **dropped** (fraud rows are cancelled, so balances leak the label), time-ordered split, no SMOTE |

The loaders are covered by tests that use tiny mock files in the exact Kaggle formats (`tests/test_loaders.py`),
but they have **not** been run on the real files in this environment (no Kaggle access). Expect to fix small schema surprises on first run.
