import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

DATA_DIR = "data"
OUT_DIR = "outputs"

STATE_NAME = "FLORIDA"
START_YEAR = 2016
END_YEAR = 2025

# forecasting split
TRAIN_END_YEAR = 2023
TEST_YEAR = 2024
FORECAST_YEAR = 2025

# classification label
HIGH_DAMAGE_Q = 0.90  # top 10%
USE_TEXT_FOR_NB = True  # set False if you do not want text

def dollars(x):
    # turns "250K", "3M", "1.2B" into a number
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "" or s == "0":
        return 0.0

    mult = 1.0
    last = s[-1].upper()
    if last == "K":
        mult = 1e3
        s = s[:-1]
    elif last == "M":
        mult = 1e6
        s = s[:-1]
    elif last == "B":
        mult = 1e9
        s = s[:-1]

    try:
        return float(s) * mult
    except ValueError:
        return 0.0

def mae_rmse(y_true, y_pred):
    e = (y_true - y_pred).astype(float)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    return mae, rmse

def load_florida():
    # searches for NOAA details files
    pats = [
        os.path.join(DATA_DIR, "StormEvents_details-ftp_*.csv"),
        os.path.join(DATA_DIR, "StormEvents_details-ftp_*.csv.gz"),
    ]
    files = []
    for p in pats:
        files += sorted(glob.glob(p))
    if not files:
        raise FileNotFoundError(
            "No StormEvents_details files found in data/. Put the NOAA .csv.gz files in data/."
        )

    keep = {
        "BEGIN_DATE_TIME",
        "STATE",
        "EVENT_TYPE",
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "EVENT_NARRATIVE",
    }

    parts = []
    for f in files:
        parts.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in keep))
    df = pd.concat(parts, ignore_index=True)

    # florida only
    df = df[df["STATE"].astype(str).str.upper() == STATE_NAME].copy()
    
    # time
    df["BEGIN_DATE_TIME"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["BEGIN_DATE_TIME"])
    df["YEAR"] = df["BEGIN_DATE_TIME"].dt.year
    df = df[(df["YEAR"] >= START_YEAR) & (df["YEAR"] <= END_YEAR)].copy()
    df["MONTH"] = df["BEGIN_DATE_TIME"].dt.month
    df["YEAR_MONTH"] = df["BEGIN_DATE_TIME"].dt.to_period("M").astype(str)

    # damage
    df["DAMAGE_PROPERTY_USD"] = df["DAMAGE_PROPERTY"].apply(dollars)
    df["DAMAGE_CROPS_USD"] = df["DAMAGE_CROPS"].apply(dollars)
    df["TOTAL_DAMAGE_USD"] = df["DAMAGE_PROPERTY_USD"] + df["DAMAGE_CROPS_USD"]

    # injuries/deaths numeric
    for c in ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["EVENT_NARRATIVE"] = df["EVENT_NARRATIVE"].fillna("")
    return df, len(files)

def monthly_series(df):
    # counts / month
    m = df.groupby("YEAR_MONTH").size().reset_index(name="EVENT_COUNT").sort_values("YEAR_MONTH")
    m["DATE"] = pd.to_datetime(m["YEAR_MONTH"] + "-01")
    y = m.set_index("DATE")["EVENT_COUNT"].asfreq("MS")
    return y

def seasonal_naive(y_train, steps):
    # repeats last year's same months
    last12 = y_train.iloc[-12:].values
    reps = int(np.ceil(steps / 12))
    vals = np.tile(last12, reps)[:steps]
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(vals, index=idx)

def forecast_models(y_all):
    # split
    y_train = y_all[(y_all.index.year >= START_YEAR) & (y_all.index.year <= TRAIN_END_YEAR)]
    y_test = y_all[y_all.index.year == TEST_YEAR]
    steps = len(y_test)

    # 3 models
    pred_naive = seasonal_naive(y_train, steps)

    pred_hw = ExponentialSmoothing(
        y_train, seasonal="add", seasonal_periods=12, trend=None
    ).fit(optimized=True).forecast(steps)

    pred_sa = SARIMAX(
        y_train,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False).forecast(steps)

    mae_n, rmse_n = mae_rmse(y_test, pred_naive)
    mae_hw, rmse_hw = mae_rmse(y_test, pred_hw)
    mae_sa, rmse_sa = mae_rmse(y_test, pred_sa)

    res = pd.DataFrame(
        [
            ["Seasonal Naive", mae_n, rmse_n],
            ["Holt-Winters", mae_hw, rmse_hw],
            ["SARIMA", mae_sa, rmse_sa],
        ],
        columns=["Model", "MAE", "RMSE"],
    ).sort_values("RMSE")

    res.to_csv(os.path.join(OUT_DIR, "forecasting_metrics_2024.csv"), index=False)

    best_name = res.iloc[0]["Model"]
    best_test_pred = {"Seasonal Naive": pred_naive, "Holt-Winters": pred_hw, "SARIMA": pred_sa}[best_name]

    # fig1 full history
    plt.figure()
    plt.plot(y_all.index, y_all.values)
    plt.title("Florida Monthly Storm Events (2016–2025)")
    plt.xlabel("Month")
    plt.ylabel("Event Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_monthly_counts_2016_2025.png"), dpi=200)
    plt.close()

    # fig2 test actual vs pred.
    plt.figure()
    plt.plot(y_train.index, y_train.values, label="Train (2016–2023)")
    plt.plot(y_test.index, y_test.values, label="Test Actual (2024)")
    plt.plot(best_test_pred.index, best_test_pred.values, label=f"Test Predicted (Best: {best_name})")
    plt.title("Forecasting Test Performance (2024)")
    plt.xlabel("Month")
    plt.ylabel("Event Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_test_actual_vs_pred_2024.png"), dpi=200)
    plt.close()

    # forecast 2025 from data thru 2024
    y_up_to_2024 = y_all[y_all.index.year <= TEST_YEAR]
    steps_2025 = int(np.sum(y_all.index.year == FORECAST_YEAR))
    if steps_2025 == 0:
        steps_2025 = 12

    if best_name == "Seasonal Naive":
        pred_2025 = seasonal_naive(y_up_to_2024, steps_2025)
    elif best_name == "Holt-Winters":
        pred_2025 = ExponentialSmoothing(
            y_up_to_2024, seasonal="add", seasonal_periods=12, trend=None
        ).fit(optimized=True).forecast(steps_2025)
    else:
        pred_2025 = SARIMAX(
            y_up_to_2024,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False).forecast(steps_2025)

    pred_2025.to_csv(os.path.join(OUT_DIR, "forecast_2025_best_model.csv"), header=["PRED_EVENT_COUNT"])

    # fig3 forecast overlay
    plt.figure()
    plt.plot(y_all.index, y_all.values, label="History (2016–2025, observed)")
    plt.plot(pred_2025.index, pred_2025.values, label=f"Forecast (Best: {best_name})")
    plt.title("Forecast for 2025 (Monthly Storm Event Counts)")
    plt.xlabel("Month")
    plt.ylabel("Event Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_forecast_2025_overlay.png"), dpi=200)
    plt.close()

    return res, pred_2025

def naive_bayes(df):
    # builds table and label
    events = df[
        [
            "MONTH",
            "EVENT_TYPE",
            "INJURIES_DIRECT",
            "INJURIES_INDIRECT",
            "DEATHS_DIRECT",
            "DEATHS_INDIRECT",
            "TOTAL_DAMAGE_USD",
            "EVENT_NARRATIVE",
        ]
    ].copy()

    thr = float(events["TOTAL_DAMAGE_USD"].quantile(HIGH_DAMAGE_Q))
    events["HIGH_DAMAGE"] = (events["TOTAL_DAMAGE_USD"] >= thr).astype(int)
    events.to_csv(os.path.join(OUT_DIR, "event_table_with_high_damage.csv"), index=False)

    X = events.copy()
    y = X.pop("HIGH_DAMAGE").astype(int)

    cat = ["EVENT_TYPE", "MONTH"]
    num = ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]

    cols = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ]
    if USE_TEXT_FOR_NB:
        cols.append(("txt", TfidfVectorizer(min_df=3, stop_words="english"), "EVENT_NARRATIVE"))

    pre = ColumnTransformer(cols)
    model = Pipeline([("pre", pre), ("nb", MultinomialNB())])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    cm = confusion_matrix(y_te, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)

    # text report
    rep = classification_report(y_te, pred, zero_division=0)
    with open(os.path.join(OUT_DIR, "naive_bayes_report.txt"), "w") as f:
        f.write(rep)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write(f"\n\nPrecision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\n")

    # fig4 confusion matrix
    plt.figure()
    plt.imshow(cm)
    plt.title("Naive Bayes Confusion Matrix (HIGH_DAMAGE)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_nb_confusion_matrix.png"), dpi=200)
    plt.close()

    # fig5 high damage rate by event type (top 10 by count)
    top10 = events["EVENT_TYPE"].value_counts().head(10).index
    tmp = events[events["EVENT_TYPE"].isin(top10)].copy()
    rate = tmp.groupby("EVENT_TYPE")["HIGH_DAMAGE"].mean().sort_values(ascending=False)

    plt.figure()
    plt.bar(rate.index, rate.values)
    plt.title("HIGH_DAMAGE Rate by Event Type (Top 10 by frequency)")
    plt.xlabel("Event Type")
    plt.ylabel("High Damage Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig5_high_damage_rate_by_event_type.png"), dpi=200)
    plt.close()

    return thr, float(p), float(r), float(f1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df, nfiles = load_florida()

    # small summary file
    with open(os.path.join(OUT_DIR, "data_summary.txt"), "w") as f:
        f.write(f"Florida rows (events): {len(df)}\n")
        f.write(f"Years: {START_YEAR}-{END_YEAR}\n")
        f.write(f"Files read: {nfiles}\n")

    # monthly series
    y = monthly_series(df)
    y.to_csv(os.path.join(OUT_DIR, "monthly_event_count_series.csv"), header=["EVENT_COUNT"])

    # forecasting
    fore_res, _ = forecast_models(y)

    # naive bayes
    thr, p, r, f1 = naive_bayes(df)

    # combined summary
    summary = {
        "best_forecasting_model": fore_res.iloc[0]["Model"],
        "best_forecasting_rmse_2024": float(fore_res.iloc[0]["RMSE"]),
        "high_damage_threshold_usd_top10pct": float(thr),
        "nb_precision": float(p),
        "nb_recall": float(r),
        "nb_f1": float(f1),
        "use_text_for_nb": USE_TEXT_FOR_NB,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "project_summary.csv"), index=False)

    print("Done.")
    print("All charts and tables saved in outputs/.")

if __name__ == "__main__":
    main()