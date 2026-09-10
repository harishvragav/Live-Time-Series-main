# 📈 Time Series Analysis & Forecasting Project

A hands-on time series analysis project that walks through **Exploratory Data Analysis (EDA)**, **stationarity testing**, and **forecasting** using **ARIMA** and **SARIMA** models — built to understand how real-world time-dependent data behaves and how to predict its future values.

---

## 🎯 Objective

The goal of this project is to:

1. **Understand how time series data behaves** — trends, seasonality, and patterns hidden in data that changes over time.
2. **Learn how to prepare time series data** for modeling (making it "stationary").
3. **Build and compare forecasting models** (ARIMA vs SARIMA) and pick the right one based on the data.
4. **Evaluate model performance** using visual and statistical methods.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `Time Series EDA.ipynb` | Exploratory Data Analysis on Tesla (TSLA) stock data |
| `ARIMA.ipynb` | ARIMA & SARIMA modeling on Airline Passengers dataset |
| `airline_passengers.csv` | Classic Box-Jenkins airline passenger dataset (1949–1960) |

---

## 📊 Datasets Used

### 1. Tesla Stock Data (TSLA)
- **Source:** Yahoo Finance (`yfinance` library)
- **Period:** January 2015 – January 2024
- **Features:** Open, High, Low, Close, Volume (daily)
- **Purpose:** Used for EDA — understanding trends, resampling, and rolling statistics.

### 2. Airline Passengers Dataset
- **Source:** Classic Box-Jenkins dataset (`airline_passengers.csv`)
- **Period:** January 1949 – December 1960 (144 monthly observations)
- **Feature:** Thousands of Passengers (monthly total)
- **Purpose:** Used for building and evaluating ARIMA and SARIMA forecasting models.

---

## 🔍 Steps Performed

### Notebook 1: Time Series EDA (`Time Series EDA.ipynb`)

This notebook focuses on **exploring and understanding** time series data before any modeling:

| Step | What Was Done | Why |
|------|---------------|-----|
| **Data Collection** | Downloaded Tesla stock data using `yfinance` | To get real-world, high-frequency time series data |
| **Basic Exploration** | `head()`, `tail()`, `info()`, null checks | To understand data shape, types, and missing values |
| **Line Plot** | Plotted the `High` price over time | To visually identify **trends** (upward/downward movements) |
| **Custom Visualization** | Applied `xlim`, `ylim`, line styles, colors | To zoom into specific periods and highlight patterns |
| **DateTime Indexing** | Set `Date` as the DataFrame index | Essential step — most time series operations require a datetime index |
| **DateTime Operations** | Extracted `day`, `month`, `year`, `weekday` | To understand datetime manipulation in Python |
| **Time Resampling** | Resampled data by Year (`YE`), Quarter (`QS`), Month (`ME`), Business Year (`BYE`) | To aggregate data at different time granularities — useful for spotting long-term trends |
| **Resampled Plots** | Bar charts of yearly mean `Open` and monthly max `Open` | To visualize how aggregated data reveals overall direction |
| **Rolling Statistics** | Calculated 30-day rolling mean of `Open` price | To smooth out short-term noise and see the **underlying trend** |
| **Rolling Plot** | Plotted `Open` vs `Open: 30 days rolling` together | To compare raw data with smoothed trend — the rolling mean reduces volatility |

### Notebook 2: ARIMA & SARIMA Modeling (`ARIMA.ipynb`)

This notebook focuses on **building forecasting models** step by step:

| Step | What Was Done | Why |
|------|---------------|-----|
| **Data Loading & Cleaning** | Loaded airline dataset, dropped NaN rows, converted `Month` to datetime, set as index | Clean data with proper datetime index is required for time series models |
| **Initial Plot** | Plotted passenger count over time | Revealed a clear **upward trend** and **seasonal pattern** (repeating peaks every 12 months) |
| **ADF Test (Original)** | Augmented Dickey-Fuller test on raw data | **p-value = 0.99** — Data is **non-stationary** (has trend/seasonality). Models like ARIMA require stationary data |
| **1st Differencing** | Subtracted each value from the previous month's value | Removes the **trend** component |
| **ADF Test (1st Diff)** | ADF test on 1st differenced data | **p-value = 0.054** — Still non-stationary at 5% significance level |
| **2nd Differencing** | Applied differencing again on 1st differenced data | Further removes trend to achieve stationarity |
| **ADF Test (2nd Diff)** | ADF test on 2nd differenced data | **p-value ≈ 2.73×10⁻²⁹** — ✅ **Stationary!** This tells us `d=2` for ARIMA |
| **Seasonal Differencing (lag=12)** | Subtracted each value from the value 12 months ago | Removes the **seasonal** component (yearly pattern) |
| **ADF Test (Seasonal)** | ADF test on seasonally differenced data | **p-value = 0.012** — ✅ **Stationary!** This tells us seasonal differencing with period `m=12` works |
| **ACF & PACF Plots** | Plotted Autocorrelation and Partial Autocorrelation for both 2nd and seasonal differences | These plots help determine the `p` (AR) and `q` (MA) parameters for the models |
| **Train/Test Split** | Train: 1949–1955, Test: 1956–1960 | To evaluate model predictions on unseen data |
| **ARIMA Model** | Built ARIMA(8, 2, 4) | Used 2nd differenced ACF/PACF to determine `p=8`, `d=2`, `q=4` |
| **ARIMA Predictions** | Predicted on test period and plotted actual vs predicted | To visually assess forecast accuracy |
| **Residual Analysis (ARIMA)** | Plotted residual KDE (density) | To check if residuals are approximately normally distributed (good model → bell curve centered at 0) |
| **SARIMA Model** | Built SARIMAX(3, 0, 5) × (0, 1, 0, 12) | Used seasonal differenced ACF/PACF: `p=3`, `d=0`, `q=5`, seasonal `D=1`, `m=12` |
| **SARIMA Predictions** | Predicted on test period | To compare with ARIMA |
| **Residual Analysis (SARIMA)** | Plotted residual time series and KDE | To evaluate SARIMA model quality |
| **Comparison Plot** | Plotted Actual vs ARIMA Predicted vs SARIMA Predicted | To visually compare which model captures the data pattern better |

---

## 🤖 ARIMA vs SARIMA — When to Use Which?

### What is ARIMA?
**ARIMA** stands for **A**uto**R**egressive **I**ntegrated **M**oving **A**verage. Think of it as a model that learns from the **recent past** to predict the future.

It has 3 parameters — **ARIMA(p, d, q)**:
| Parameter | Meaning | How to Find It |
|-----------|---------|----------------|
| **p** (AutoRegressive order) | How many past values influence the current value | Look at the **PACF plot** — count significant lags |
| **d** (Differencing order) | How many times to subtract consecutive values to remove the trend | Use the **ADF test** — keep differencing until p-value < 0.05 |
| **q** (Moving Average order) | How many past forecast errors influence the current value | Look at the **ACF plot** — count significant lags |

**Use ARIMA when:** Your data has a **trend** but **no repeating seasonal pattern** (e.g., stock prices, GDP growth).

### What is SARIMA?
**SARIMA** is **Seasonal ARIMA**. It extends ARIMA by adding seasonal components to capture **repeating patterns** (like summer peaks in air travel every year).

It has 7 parameters — **SARIMA(p, d, q) × (P, D, Q, m)**:
| Parameter | Meaning |
|-----------|---------|
| **p, d, q** | Same as ARIMA (non-seasonal part) |
| **P** | Seasonal autoregressive order |
| **D** | Seasonal differencing order (usually 1) |
| **Q** | Seasonal moving average order |
| **m** | Season length (12 for monthly data with yearly seasonality) |

**Use SARIMA when:** Your data has **both a trend AND a seasonal pattern** (e.g., airline passengers, monthly retail sales, electricity demand).

### In This Project
- **ARIMA(8, 2, 4):** Captured the overall trend but **struggled with the seasonal ups and downs** because it doesn't understand repeating yearly patterns.
- **SARIMA(3, 0, 5) × (0, 1, 0, 12):** **Much better fit** because it explicitly models the 12-month seasonal cycle.

> **Rule of Thumb:** If your data shows a repeating pattern at regular intervals (weekly, monthly, yearly), always prefer **SARIMA** over plain ARIMA.

---

## 📏 Evaluation Metrics & Model Assessment

Models were evaluated using:

| Method | What It Tells You |
|--------|-------------------|
| **ADF Test (Augmented Dickey-Fuller)** | Whether the data is stationary. A **p-value < 0.05** means stationary (good for modeling) |
| **ACF Plot (Autocorrelation Function)** | Shows correlation of the series with its own lagged values — helps pick `q` parameter |
| **PACF Plot (Partial Autocorrelation Function)** | Shows direct correlation with each lag (removing intermediate effects) — helps pick `p` parameter |
| **Residual Plot** | Time series plot of prediction errors. Ideally should look like **random noise** (no patterns) |
| **Residual KDE (Kernel Density Estimate)** | Density plot of residuals. A good model produces residuals shaped like a **bell curve centered at zero** |
| **Actual vs Predicted Plot** | The most intuitive check — how closely the predicted line follows the actual data |

---

## 📈 Key Visualizations Explained

| Plot | What It Shows |
|------|---------------|
| **Raw Time Series Plot** | Shows overall trend and seasonality in the data |
| **Differenced Series Plot** | Shows data after removing trend — should look like random fluctuations if stationary |
| **ACF Plot** | Bars showing correlation at each lag — bars outside the shaded confidence band are "significant" |
| **PACF Plot** | Similar to ACF but shows "direct" effect of each lag — helps choose the AR order (p) |
| **Actual vs Predicted** | Overlays model predictions on real data — closer = better model |
| **Residual KDE** | Bell-shaped curve centered at 0 = good model; skewed/wide = model missing patterns |

---

## 🛠️ Tech Stack

- **Python 3.10**
- **pandas** — Data manipulation and time series operations
- **numpy** — Numerical computing
- **matplotlib** — Visualization
- **statsmodels** — ARIMA, SARIMA, ADF test, ACF/PACF plots
- **yfinance** — Fetching live stock market data

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/harishvragav/Live-Time-Series-main.git
   cd Live-Time-Series-main
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib statsmodels yfinance
   ```

3. Open the notebooks:
   ```bash
   jupyter notebook
   ```

4. Run the notebooks in order:
   - First: `Time Series EDA.ipynb` (understand the data)
   - Then: `ARIMA.ipynb` (build and compare models)

---

## 📜 License

This project is licensed under the **GPL-2.0 License** — see the [LICENSE](LICENSE) file for details.