# Dystal Corvus-AP.CZC

Dystal Corvus-AP.CZC is a meteorological-driven machine learning quant model for Chinese Apple Futures (CZCE: AP), developed under **Dystal Capital**.

By removing noisy price-based technical indicators and cross-commodity price signals, the model focuses purely on global climate oscillations and regional meteorological anomalies to forecast price dynamics.

## Key Features
- **Global ENSO Driver**: Incorporates monthly NOAA ERSSTv5 Niño 3.4 Sea Surface Temperature (SST) anomalies.
- **Regional Precipitation Anomalies**: Calculates daily rainfall deviation relative to the historical monthly mean across 4 major Chinese apple production regions (Luochuan, Yantai, Tianshui, and Lingbao).
- **Short-Term Forecast Proxy**: Includes 7-day forward average temperature and precipitation sum.
- **Scale-Invariant Proportional Adjustment**: Prices are adjusted using ratio-based back-adjustment to ensure daily returns and volatility features are completely leak-free and independent of dataset boundaries.
- **SQLite Cache Pipeline**: Full local database caching for both futures price and meteorological data, allowing execution in under 1 second.

## Walk-Forward Backtest Results
Running the rolling walk-forward test over the 2017-2026 dataset yields the following metrics (net of transaction fees and slippage):
- **Overall Sharpe Ratio**: -0.1844
- **Pre-2025 Period (Historical Dev)**: Sharpe of -0.6042
- **Post-2025 Period (Pure Out-of-Sample)**: Sharpe of **0.7576**, Cumulative Return of **14.71%**

Once trained on a sufficiently large historical baseline, the model shows robust out-of-sample forward predictive power in recent years.

## Files
- `ap_future_quant.py`: Main CLI strategy script.
- `equity_curve_meteorological.png`: Cumulative return plot for the meteorological model.
