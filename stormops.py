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


# -----------------------------
# Project settings (easy to explain)
# -----------------------------
DATA_DIR = "data"
OUT_DIR = "outputs"

STATE_NAME = "FLORIDA"
START_YEAR = 2016
END_YEAR = 2025

TRAIN_END_YEAR = 2023
TEST_YEAR = 2024
FORECAST_YEAR = 2025

HIGH_DAMAGE_Q = 0.90  # top 10% damage
USE_TEXT_FOR_NB = True  # set False if you want no text features


# -----------------------------
# Small helper functions
# -----------------------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def parse_damage(value) -> float:
    """
    NOAA damage fields look like '250K', '3M', '1.2B', blank.
    Convert to dollars as float.
    """
    if pd.isna(value):
        return 0.0

    s = str(value).strip()
    if s == "" or s == "0":
        return 0.0

    last = s[-1].upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}

    if last in multipliers:
        num_part = s[:-1]
        mult = multipliers[last]
    else:
        num_part = s
        mult = 1.0

    try:
        return float(num_part) * mult
    except ValueError:
        return 0.0


def mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    err = (y_true - y_pred).astype(float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


# -----------------------------
# Data loading and preprocessing
# -----------------------------
def find_noaa_files() -> list[str]:
    patterns = [
        os.path.join(DATA_DIR, "StormEvents_details-ftp_*.csv"),
        os.path.join(DATA_DIR, "StormEvents_details-ftp_*.csv.gz"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    return files


def load_florida_noaa_details(files: list[str]) -> pd.DataFrame:
    """
    Loads all details files.
    Keeps only needed columns.
    Filters Florida and years 2016-2025.
    Creates numeric damage fields and time features.
    """
    if not files:
        raise FileNotFoundError(
            "No StormEvents_details files found in data/. "
            "Put the NOAA .csv.gz files in the data/ folder."
        )

    keep_cols = {
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

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, usecols=lambda c: c in keep_cols)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    # Filter Florida
    fl = raw[raw["STATE"].astype(str).str.upper() == STATE_NAME].copy()

    # Parse datetime and filter years
    fl["BEGIN_DATE_TIME"] = pd.to_datetime(fl["BEGIN_DATE_TIME"], errors="coerce")
    fl = fl.dropna(subset=["BEGIN_DATE_TIME"])
    fl["YEAR"] = fl["BEGIN_DATE_TIME"].dt.year
    fl = fl[(fl["YEAR"] >= START_YEAR) & (fl["YEAR"] <= END_YEAR)].copy()

    # Month and year-month key
    fl["MONTH"] = fl["BEGIN_DATE_TIME"].dt.month
    fl["YEAR_MONTH"] = fl["BEGIN_DATE_TIME"].dt.to_period("M").astype(str)

    # Damage parsing
    fl["DAMAGE_PROPERTY_USD"] = fl["DAMAGE_PROPERTY"].apply(parse_damage)
    fl["DAMAGE_CROPS_USD"] = fl["DAMAGE_CROPS"].apply(parse_damage)
    fl["TOTAL_DAMAGE_USD"] = fl["DAMAGE_PROPERTY_USD"] + fl["DAMAGE_CROPS_USD"]

    # Numeric injury/death fields
    num_cols = ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]
    for c in num_cols:
        fl[c] = pd.to_numeric(fl[c], errors="coerce").fillna(0.0)

    fl["EVENT_NARRATIVE"] = fl["EVENT_NARRATIVE"].fillna("")

    return fl


def build_monthly_series(fl: pd.DataFrame) -> pd.Series:
    """
    Monthly event count series with a proper monthly DateTimeIndex.
    """
    monthly = (
        fl.groupby("YEAR_MONTH")
        .size()
        .reset_index(name="EVENT_COUNT")
        .sort_values("YEAR_MONTH")
    )
    monthly["DATE"] = pd.to_datetime(monthly["YEAR_MONTH"] + "-01")
    y = monthly.set_index("DATE")["EVENT_COUNT"].asfreq("MS")
    return y


# -----------------------------
# Forecasting models (3 required)
# -----------------------------
def seasonal_naive(y_train: pd.Series, steps: int) -> pd.Series:
    last12 = y_train.iloc[-12:].values
    reps = int(np.ceil(steps / 12))
    values = np.tile(last12, reps)[:steps]
    idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(values, index=idx)


def holt_winters(y_train: pd.Series, steps: int) -> pd.Series:
    model = ExponentialSmoothing(
        y_train,
        seasonal="add",
        seasonal_periods=12,
        trend=None,
    ).fit(optimized=True)
    return model.forecast(steps)


def sarima(y_train: pd.Series, steps: int) -> pd.Series:
    model = SARIMAX(
        y_train,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model.forecast(steps)


def run_forecasting(y_all: pd.Series) -> pd.DataFrame:
    """
    Train: 2016-2023
    Test: 2024
    Forecast: 2025
    """
    y_train = y_all[(y_all.index.year >= START_YEAR) & (y_all.index.year <= TRAIN_END_YEAR)]
    y_test = y_all[y_all.index.year == TEST_YEAR]

    steps_test = len(y_test)

    pred_naive = seasonal_naive(y_train, steps_test)
    pred_hw = holt_winters(y_train, steps_test)
    pred_sa = sarima(y_train, steps_test)

    mae_n, rmse_n = mae_rmse(y_test, pred_naive)
    mae_hw, rmse_hw = mae_rmse(y_test, pred_hw)
    mae_sa, rmse_sa = mae_rmse(y_test, pred_sa)

    results = pd.DataFrame(
        [
            ["Seasonal Naive", mae_n, rmse_n],
            ["Holt-Winters", mae_hw, rmse_hw],
            ["SARIMA", mae_sa, rmse_sa],
        ],
        columns=["Model", "MAE", "RMSE"],
    ).sort_values("RMSE")

    results.to_csv(os.path.join(OUT_DIR, "forecasting_metrics_2024.csv"), index=False)

    # Best model chosen by lowest RMSE
    best_model = results.iloc[0]["Model"]
    best_pred_test = {
        "Seasonal Naive": pred_naive,
        "Holt-Winters": pred_hw,
        "SARIMA": pred_sa,
    }[best_model]

    # Chart 1: Full history (counts)
    plt.figure()
    plt.plot(y_all.index, y_all.values)
    plt.title("Florida Monthly Storm Events (2016–2025)")
    plt.xlabel("Month")
    plt.ylabel("Event Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_monthly_counts_2016_2025.png"), dpi=200)
    plt.close()

    # Chart 2: Actual vs predicted on 2024
    plt.figure()
    plt.plot(y_train.index, y_train.values, label="Train (2016–2023)")
    plt.plot(y_test.index, y_test.values, label="Test Actual (2024)")
    plt.plot(best_pred_test.index, best_pred_test.values, label=f"Test Predicted (Best: {best_model})")
    plt.title("Forecasting Test Performance (2024)")
    plt.xlabel("Month")
    plt.ylabel("Event Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_test_actual_vs_pred_2024.png"), dpi=200)
    plt.close()

    # Forecast 2025 using best model trained through end of 2024
    y_through_2024 = y_all[y_all.index.year <= TEST_YEAR]
    steps_2025 = int(np.sum(y_all.index.year == FORECAST_YEAR))
    if steps_2025 == 0:
        # If 2025 not present in the data, still forecast 12 months
        steps_2025 = 12

    if best_model == "Seasonal Naive":
        pred_2025 = seasonal_naive(y_through_2024, steps_2025)
    elif best_model == "Holt-Winters":
        pred_2025 = holt_winters(y_through_2024, steps_2025)
    else:
        pred_2025 = sarima(y_through_2024, steps_2025)

    pred_2025.to_csv(os.path.join(OUT_DIR, "forecast_2025_best_model.csv"), header=["PRED_EVENT_COUNT"])

    # Chart 3: Forecast 2025 overlay
    plt.figure()
    plt.plot(y_all.index, y_all.values, label="History (2016–2025, observed)")
    plt.plot(pred_2025.index, pred_2025.values, label=f"Forecast (Best: {best_model})")
    plt.title("Forecast for 2025 (Monthly Storm Event Counts)")
    plt.xlabel("Month")
    plt.ylabel("Event Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_forecast_2025_overlay.png"), dpi=200)
    plt.close()

    return results


# -----------------------------
# Naive Bayes classification (damage triage)
# -----------------------------
def build_event_table(fl: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Label HIGH_DAMAGE using top 10% by TOTAL_DAMAGE_USD.
    """
    events = fl[
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

    threshold = events["TOTAL_DAMAGE_USD"].quantile(HIGH_DAMAGE_Q)
    events["HIGH_DAMAGE"] = (events["TOTAL_DAMAGE_USD"] >= threshold).astype(int)

    events.to_csv(os.path.join(OUT_DIR, "event_table_with_high_damage.csv"), index=False)
    return events, float(threshold)


def run_naive_bayes(events: pd.DataFrame) -> dict:
    """
    MultinomialNB with:
      categorical one-hot (EVENT_TYPE, MONTH)
      numeric passthrough (injuries, deaths)
      optional TF-IDF text (EVENT_NARRATIVE)
    """
    X = events.copy()
    y = X.pop("HIGH_DAMAGE").astype(int)

    categorical = ["EVENT_TYPE", "MONTH"]
    numeric = ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]

    if USE_TEXT_FOR_NB:
        transformers.append(("txt", TfidfVectorizer(min_df=3, stop_words="english"), "EVENT_NARRATIVE"))

    pre = ColumnTransformer(transformers)

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("nb", MultinomialNB()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    # Save report
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(OUT_DIR, "naive_bayes_report.txt"), "w") as f:
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nPrecision: {:.4f}\nRecall: {:.4f}\nF1: {:.4f}\n".format(p, r, f1))

    # Chart 4: Confusion matrix image
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

    # Chart 5: High-damage rate by event type (top 10 event types by volume)
    tmp = events.copy()
    counts = tmp["EVENT_TYPE"].value_counts()
    top_types = counts.head(10).index
    tmp = tmp[tmp["EVENT_TYPE"].isin(top_types)]
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

    return {"precision": float(p), "recall": float(r), "f1": float(f1), "cm": cm.tolist()}


# -----------------------------
# Main runner
# -----------------------------
def main():
    ensure_dirs()

    files = find_noaa_files()
    fl = load_florida_noaa_details(files)

    # Save a simple summary for slides
    with open(os.path.join(OUT_DIR, "data_summary.txt"), "w") as f:
        f.write(f"Florida rows (events): {len(fl)}\n")
        f.write(f"Years: {START_YEAR}-{END_YEAR}\n")
        f.write(f"Files read: {len(files)}\n")

    # Build monthly series and run forecasting
    y_all = build_monthly_series(fl)
    y_all.to_csv(os.path.join(OUT_DIR, "monthly_event_count_series.csv"), header=["EVENT_COUNT"])

    forecasting_results = run_forecasting(y_all)

    # Naive Bayes
    events, damage_thr = build_event_table(fl)
    nb_metrics = run_naive_bayes(events)

    # Save one combined summary table
    summary = {
        "best_forecasting_model": forecasting_results.iloc[0]["Model"],
        "best_forecasting_rmse_2024": float(forecasting_results.iloc[0]["RMSE"]),
        "high_damage_threshold_usd_top10pct": float(damage_thr),
        "nb_precision": nb_metrics["precision"],
        "nb_recall": nb_metrics["recall"],
        "nb_f1": nb_metrics["f1"],
        "use_text_for_nb": USE_TEXT_FOR_NB,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "project_summary.csv"), index=False)

    print("Done.")
    print("All charts and tables saved in outputs/.")


if __name__ == "__main__":
    main()