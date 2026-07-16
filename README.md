# German Electricity Demand — Time-Series Forecasting

A case study forecasting **German national electricity load** (Open Power System Data) using a
progression of models — classical benchmarks, SARIMA / SARIMAX, a feature-based Random Forest, and an
LSTM — compared on a **2-year (104-week) hold-out horizon**.

---

## Repository contents

| File | Description |
|------|-------------|
| [`Sony_Final_Code.ipynb`](Sony_Final_Code.ipynb) | Full analysis notebook (EDA → benchmarks → SARIMA/SARIMAX → RF → LSTM → Parts 1–7 write-up). |
| `requirements.txt` | Pinned Python dependencies. |
| `README.md` | This file. |

---

## 1. Data

- **Source:** [Open Power System Data — time series](https://data.open-power-system-data.org/time_series/) (60-minute file, `2020-10-06`). Download the file from here **or** use the copy already bundled in this repo (`opsd_60min_raw.csv`).
- **Series:** `DE_load_actual_entsoe_transparency` (German actual load, MW).
- **Window:** 1 Jan 2015 → 30 Sep 2020, aggregated to **daily** and **weekly** means (50,400 hourly → 301 weekly observations).
- **Weekly load:** mean 55,484.35 MW (σ = 3,762.74 MW), ranging from 46,505.31 MW (summer troughs) to 63,587.01 MW (winter peaks).
- **Exogenous data:** Berlin 2 m temperature from the [Open-Meteo archive API](https://archive-api.open-meteo.com/v1/archive) (fetched live — needs internet) and German public holidays via the `holidays` package (offline, no internet needed).

> **Reproducibility:** the notebook looks for `opsd_60min_raw.csv` in the same folder as
> `Sony_Final_Code.ipynb`. If it isn't there, the first cell downloads it directly from the OPSD link
> above and saves a local copy automatically — the dataset itself is **not** committed to this repo
> (it's a large file, well over GitHub's 100MB per-file limit), so the auto-download is what makes the
> notebook runnable out of the box. First run needs internet for this download and for the Open-Meteo
> temperature call later on.

---

## 2. Pipeline (assignment Parts 1–7)

| Part | Content |
|------|---------|
| 1 | Data prep + EDA: daily/weekly aggregation, seasonal decomposition, ADF/KPSS stationarity, ACF/PACF |
| 2 | Benchmarks: Mean, Naive, **Seasonal Naive**, Drift |
| 3 | **SARIMA**: grid search `p∈[0,6]`, `d∈[0,2]`, `q∈[0,6]` (147 combinations); valid within-`d` AIC + parsimony; seasonal `(P,Q)` search on `{0,1}` with `D=1` at `s=52`; residual diagnostics |
| 4 | **SARIMAX**: temperature, temperature², 1-week temperature lag, holiday flag (conditional forecast) |
| 5 | **Random Forest**: recursive multi-step forecast (comparable to SARIMA) **and** a 1-step reference |
| 6 | **LSTM** (hourly): 5-config hyperparameter search, rolling vs open-loop evaluation |
| 7 | Analytical answers to the 6 assignment questions + consolidated RMSE / MAE / MAPE comparison |

### Methodological notes
- Training runs through **7 Oct 2018**; the 104-week test window runs to **4 Oct 2020** (197 train / 104 test weeks).
- **AIC is compared only within a fixed differencing order** — comparing across `d` is invalid because the likelihood is computed on the differenced series. `d=1` is justified from the stationarity tests (`d=2` over-differences).
- **Residual diagnostics** use standardized residuals with the state-space burn-in removed.
- **Fair comparison:** 1-step forecasts (which consume the actual previous week) are labelled separately and never compared with multi-step forecasts.
- Every model reports **RMSE, MAE and MAPE**.

---

## 3. Results (weekly hold-out, 104 weeks)

**Multi-step forecasts** (single forecast origin — the fair comparison):

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | Type |
|-------|----------:|---------:|---------:|------|
| **Seasonal Naive** | **3006.8** | 2318.5 | 4.41 | multi-step (best) |
| Random Forest (recursive) | 3049.8 | 2340.8 | 4.49 | multi-step (conditional) |
| SARIMAX (Temp + Holiday) | 3418.0 | 2682.5 | 5.12 | multi-step (conditional) |
| SARIMA | 3835.7 | 3113.6 | 5.91 | multi-step |
| Mean | 4397.3 | 3788.8 | 6.97 | multi-step |
| Naive | 4459.1 | 3783.2 | 6.79 | multi-step |
| LSTM (open-loop) | 4606.8 | 3967.5 | 7.41 | multi-step |
| Drift | 5118.0 | 4339.9 | 8.05 | multi-step |

**1-step / walk-forward** (reference only — see the actual previous value, so **not** comparable to the above):

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | Type |
|-------|----------:|---------:|---------:|------|
| LSTM (rolling) | 253.1 | 206.0 | 0.39 | 1-step (uses actual lag-1) |
| Random Forest (1-step) | 2551.5 | 1841.9 | 3.54 | 1-step (uses actual lag-1) |

Selected SARIMA model: **SARIMA(1,1,6)×(0,1,1)₅₂** (AIC = 1489.65; the seasonal search dropped the insignificant seasonal AR term).

Best LSTM architecture (5-config search): **128/64-unit stacked LSTM**, dropout 0.3, batch size 64 — lowest validation loss and lowest hourly test RMSE (1,060.32 MW) of all configurations tried.

### Key findings
- **No model beats the Seasonal Naive benchmark on multi-step accuracy.** German weekly demand is dominated by a stable annual cycle, so "repeat last year" is a very strong 2-year baseline; the recursive Random Forest essentially matches it (+43 MW, ≈1.4%).
- The **2020 COVID-19 demand dip** falls in the test window — an exogenous shock none of the models anticipate.
- **1-step** RF / rolling LSTM have low RMSE only because they see the actual previous value; they are not comparable to multi-step forecasts.
- **Recommended for operational use: SARIMAX** — not for top accuracy, but for native confidence intervals, interpretable temperature/holiday coefficients, exogenous-driver support, and low maintenance.
- SARIMA/SARIMAX residuals are **not perfect white noise** (significant Ljung-Box, non-normal per Shapiro-Wilk), so the Gaussian confidence intervals are approximate.

---

## 4. How to run

**Environment:** Python 3.12 (see `requirements.txt`).

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch Jupyter and run the notebook top to bottom
jupyter notebook Sony_Final_Code.ipynb
```

**Prerequisites**
- **`opsd_60min_raw.csv` must be in the same folder as the notebook** — if not it can fetch from the OPSD.
- **Internet enabled** on first run for the Open-Meteo temperature API call.
- **GPU optional** — speeds up the LSTM stage, but it's only 10 epochs with early stopping, so CPU works fine too.

**Runtime:** roughly 30–60 minutes, dominated by the SARIMA grid search (147 candidate fits) and the LSTM open-loop recursive forecast (17,472 sequential steps).

---

## 5. References

- Open Power System Data — Time series (Germany, `DE`).
- Open-Meteo Historical Weather (Archive) API.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*, 2nd ed. Springer.
- Kong, W. et al. (2017). "Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network." *IEEE Transactions on Smart Grid*.
- Seabold, S. & Perktold, J. (2010). "statsmodels." *Proc. 9th Python in Science Conf.*
- Pedregosa, F. et al. (2011). "Scikit-learn." *JMLR*, 12, 2825–2830.

---

## 6. Notes & limitations

- SARIMAX / recursive-RF use **observed** future temperature → **conditional (explanatory) forecasts**, not fully operational (a real deployment needs a weather forecast). Holidays are deterministic and known in advance.
- The open-loop LSTM is shown separately because its recursive 2-year drift would otherwise distort the comparison plot; its RMSE also varies between runs (GPU non-determinism).
- The dataset ends Oct 2020, so the "compare against data collected after the forecast period" discussion uses the held-out test window (Oct 2018 – Oct 2020), which is real data recorded after the training cut-off.
