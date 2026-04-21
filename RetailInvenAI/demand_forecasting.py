import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sns.set_theme(style="whitegrid", palette="muted")

# ── Palette (white background) ───────────────────────────────────────────────
BLUE   = "#2471A3"    # strong blue
TEAL   = "#148F77"    # forest teal
PURPLE = "#7D3C98"    # deep violet
ORANGE = "#CA6F1E"    # burnt orange
RED    = "#C0392B"    # crimson
GREEN  = "#1E8449"    # emerald green
BG     = "white"      # figure background
PANEL  = "#F4F6F9"   # subplot face (very light grey)


def load_data():
    return pd.read_csv("demand_forecasting.csv")


def train_model():
    df = load_data()

    # Encode categorical columns
    df["Promotions"] = df["Promotions"].map({"Yes": 1, "No": 0})
    df["Demand Trend"] = df["Demand Trend"].map({
        "Increasing": 2,
        "Stable": 1,
        "Decreasing": 0
    })

    X = df[["Price", "Promotions", "Demand Trend"]]
    y = df["Sales Quantity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2    = r2_score(y_test, y_pred)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    mape  = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    metrics = {
        "r2":      round(r2, 4),
        "mae":     round(mae, 2),
        "rmse":    round(rmse, 2),
        "mape":    round(mape, 2),
        "cv_mean": round(cv_mean, 4),
        "cv_std":  round(cv_std, 4),
        # store test arrays for plotting
        "_y_test": y_test.values,
        "_y_pred": y_pred,
        "_X":      X,
        "_y":      y,
    }

    return model, metrics


# Train model once at import
model, model_metrics = train_model()
model_score = model_metrics["r2"]


def predict_demand(price, promo, trend):
    promo_map = {"Yes": 1, "No": 0}
    trend_map = {"Increasing": 2, "Stable": 1, "Decreasing": 0}

    input_data = pd.DataFrame(
        [[price, promo_map[promo], trend_map[trend]]],
        columns=["Price", "Promotions", "Demand Trend"]
    )

    prediction = model.predict(input_data)
    return int(prediction[0])


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent chart function (called during CSV batch execution)
# ─────────────────────────────────────────────────────────────────────────────
def plot_agent_charts(demand_predictions: np.ndarray):
    """
    Display 4 demand-agent charts during CSV execution.

    Parameters
    ----------
    demand_predictions : array-like
        The predicted demand values for all processed products.
    """
    print("\n" + "─" * 70)
    print("  📊  DEMAND AGENT — Charts")
    print("─" * 70)

    y_test = model_metrics["_y_test"]
    y_pred = model_metrics["_y_pred"]
    X      = model_metrics["_X"]
    y      = model_metrics["_y"]

    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    fig.suptitle(
        "Demand Forecasting Agent  —  Analysis Dashboard",
        fontsize=16, fontweight="bold", color="#1A252F", y=1.02
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.45, wspace=0.38)

    def styled_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="#2C3E50", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDC3C7")
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#1A252F", pad=10)
        ax.xaxis.label.set_color("#2C3E50")
        ax.yaxis.label.set_color("#2C3E50")

    # ── 1. Actual vs Predicted scatter ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.45, color=BLUE, edgecolors="white",
                linewidths=0.3, s=35, label="Products")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], "--", color=RED, linewidth=1.5, label="Perfect Fit")
    styled_ax(ax1, f"Actual vs Predicted Sales\nR²={model_metrics['r2']}  MAE={model_metrics['mae']}")
    ax1.set_xlabel("Actual Qty"); ax1.set_ylabel("Predicted Qty")
    ax1.legend(fontsize=8)

    # ── 2. Residuals histogram ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=28, color=PURPLE, edgecolor="white", alpha=0.85)
    ax2.axvline(0, color=RED, linestyle="--", linewidth=1.6, label="Zero error")
    ax2.axvline(residuals.mean(), color=ORANGE, linestyle="-.",
                linewidth=1.3, label=f"Mean={residuals.mean():.1f}")
    styled_ax(ax2, f"Residuals Distribution\nRMSE={model_metrics['rmse']}  MAPE={model_metrics['mape']}%")
    ax2.set_xlabel("Error (Actual − Predicted)"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)

    # ── 3. Feature importance bar ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    fi = model.feature_importances_
    feats = ["Price", "Promotions", "Demand Trend"]
    colors_fi = [BLUE, TEAL, PURPLE]
    bars = ax3.barh(feats, fi, color=colors_fi, edgecolor="white", height=0.5)
    for bar, v in zip(bars, fi):
        ax3.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", color="#1A252F", fontsize=9,
                 fontweight="bold")
    styled_ax(ax3, "Feature Importances")
    ax3.set_xlabel("Importance Score")
    ax3.set_xlim(0, max(fi) * 1.2)

    plt.tight_layout()
    plt.savefig("demand_agent_charts.png", dpi=180, bbox_inches="tight",
                facecolor=BG)
    print("  [Saved] demand_agent_charts.png")
    plt.show()
