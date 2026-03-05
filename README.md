# StormOps Florida (2016–2025)

This project:
- Forecasts monthly Florida storm event counts using 3 forecasting models - Holt-Winters, SARIMA & Seasonal Naive
- Classifies high-damage events using Naive Bayes

## Dataset
NOAA Storm Events bulk CSV directory:
https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Download `StormEvents_details-ftp_...` for years 2016–2025 into `data/`.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt