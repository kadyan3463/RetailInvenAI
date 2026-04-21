import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    return pd.read_csv("pricing_optimization.csv")


def train_model():
    df = load_data()

    X = df[["Price", "Competitor Prices", "Discounts", "Elasticity Index"]]
    y = df["Sales Volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    feat_imp = dict(zip(X.columns, model.feature_importances_))

    metrics = {
        "r2":      round(r2, 4),
        "mae":     round(mae, 2),
        "rmse":    round(rmse, 2),
        "mape":    round(mape, 2),
        "cv_mean": round(cv_mean, 4),
        "cv_std":  round(cv_std, 4),
        "feat_imp": {k: round(v, 4) for k, v in feat_imp.items()},
        # store for plotting
        "_y_test": y_test.values,
        "_y_pred": y_pred,
        "_X":      X,
        "_y":      y,
    }

    return model, metrics


# Train model once at import
model, model_metrics = train_model()
model_score = model_metrics["r2"]


def predict_sales_raw(price, competitor_price, discount, elasticity):
    """Return raw float volume prediction (used internally for optimization)."""
    input_data = pd.DataFrame(
        [[price, competitor_price, discount, elasticity]],
        columns=["Price", "Competitor Prices", "Discounts", "Elasticity Index"]
    )
    return float(model.predict(input_data)[0])


def predict_sales(price, competitor_price, discount, elasticity):
    """Return integer volume prediction (public API, backward-compatible)."""
    return int(predict_sales_raw(price, competitor_price, discount, elasticity))


def predict_current_revenue(current_price, competitor_price, discount, elasticity):
    """Compute revenue at the CURRENT price using the model (not a proxy)."""
    vol = predict_sales_raw(current_price, competitor_price, discount, elasticity)
    return round(current_price * vol, 2)


def predict_optimal_price(current_price, competitor_price, discount, elasticity):
    """
    Finds the price that maximises revenue (Price × Volume)
    within a +/- 25% range of the current price using a fine grid.

    Uses raw float volume so fractional differences are not lost to flooring.
    Returns: (optimal_price, max_revenue, revenue_curve)
    """
    potential_prices = np.linspace(current_price * 0.75, current_price * 1.25, 100)

    best_price    = current_price
    max_revenue   = -np.inf
    revenue_curve = []

    for p in potential_prices:
        vol     = predict_sales_raw(p, competitor_price, discount, elasticity)
        revenue = p * vol
        revenue_curve.append((round(float(p), 2), round(vol, 2), round(revenue, 2)))

        if revenue > max_revenue:
            max_revenue = revenue
            best_price  = p

    return round(float(best_price), 2), round(float(max_revenue), 2), revenue_curve


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent chart function (called during CSV batch execution)
# ─────────────────────────────────────────────────────────────────────────────
def plot_agent_charts(current_prices:    np.ndarray,
                      optimal_prices:    np.ndarray,
                      max_revenues:      np.ndarray,
                      competitor_prices: np.ndarray,
                      discounts:         np.ndarray,
                      elasticities:      np.ndarray):
    """
    Display pricing-agent charts during CSV execution.

    Parameters
    ----------
    current_prices    : array-like – original prices for all products
    optimal_prices    : array-like – optimal prices predicted by the agent
    max_revenues      : array-like – maximum revenues at optimal price
    competitor_prices : array-like – competitor prices (needed for baseline rev)
    discounts         : array-like – discount values
    elasticities      : array-like – elasticity index values
    """
    print("\n" + "─" * 70)
    print("  💲  PRICING AGENT — Charts")
    print("─" * 70)

    y_test = model_metrics["_y_test"]
    y_pred = model_metrics["_y_pred"]
    X      = model_metrics["_X"]
    y      = model_metrics["_y"]

    current_prices    = np.asarray(current_prices,    dtype=float)
    optimal_prices    = np.asarray(optimal_prices,    dtype=float)
    max_revenues      = np.asarray(max_revenues,      dtype=float)
    competitor_prices = np.asarray(competitor_prices, dtype=float)
    discounts         = np.asarray(discounts,         dtype=float)
    elasticities      = np.asarray(elasticities,      dtype=float)

    # ── Compute REAL current revenue using the model (not a proxy) ────────────
    current_revenues = np.array([
        predict_current_revenue(cp, cmp, d, e)
        for cp, cmp, d, e in zip(current_prices, competitor_prices, discounts, elasticities)
    ])

    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    fig.suptitle(
        "Pricing Optimization Agent  —  Analysis Dashboard",
        fontsize=16, fontweight="bold", color="#1A252F", y=1.02
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    def styled_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="#2C3E50")
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDC3C7")
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#1A252F", pad=10)
        ax.xaxis.label.set_color("#2C3E50")
        ax.yaxis.label.set_color("#2C3E50")

    TOP = min(15, len(current_prices))
    idx = np.arange(TOP)
    w   = 0.35

    # ── 1. Current vs Optimal price grouped bar ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(idx - w/2, current_prices[:TOP], width=w, color=ORANGE,
            label="Current Price", edgecolor="white")
    ax1.bar(idx + w/2, optimal_prices[:TOP], width=w, color=GREEN,
            label="Optimal Price", edgecolor="white")
    styled_ax(ax1, f"Current vs Optimal Price (Top {TOP} Products)\nR²={model_metrics['r2']}")
    ax1.set_xlabel("Product #"); ax1.set_ylabel("Price ($)")
    ax1.set_xticks(idx)
    ax1.legend(fontsize=8)

    # ── 2. Feature importance ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 1])
    fi    = model_metrics["feat_imp"]
    feats = list(fi.keys())
    vals  = list(fi.values())
    bar_c = [BLUE, TEAL, PURPLE, ORANGE]
    bars = ax3.barh(feats, vals, color=bar_c, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax3.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", color="#1A252F", fontsize=9, fontweight="bold")
    styled_ax(ax3, "Feature Importances")
    ax3.set_xlabel("Importance")

    # ── 3. Revenue: model-predicted current revenue vs max revenue ───────────
    ax4 = fig.add_subplot(gs[0, 2])
    cur_rev_top = current_revenues[:TOP]
    max_rev_top = max_revenues[:TOP]
    # Revenue uplift % annotation
    pct_gain = np.where(
        cur_rev_top > 0,
        (max_rev_top - cur_rev_top) / cur_rev_top * 100,
        0
    )
    ax4.plot(idx, cur_rev_top, marker="o", color=ORANGE,
             label="Current Revenue (model)", linewidth=1.8)
    ax4.plot(idx, max_rev_top, marker="s", color=GREEN,
             label="Max Revenue @ Optimal", linewidth=1.8)
    ax4.fill_between(idx, cur_rev_top, max_rev_top, alpha=0.15, color=GREEN)
    # annotate uplift %
    for i_p, (cr, mr, pg) in enumerate(zip(cur_rev_top, max_rev_top, pct_gain)):
        mid = (cr + mr) / 2
        ax4.annotate(f"+{pg:.1f}%", (i_p, mid),
                     textcoords="offset points", xytext=(3, 0),
                     fontsize=7, color=GREEN, fontweight="bold")
    styled_ax(ax4, f"Revenue Uplift (Top {TOP} Products) — Model-Based Baseline")
    ax4.set_xlabel("Product #"); ax4.set_ylabel("Revenue ($)")
    ax4.set_xticks(idx)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("pricing_agent_charts.png", dpi=180, bbox_inches="tight",
                facecolor=BG)
    print("  [Saved] pricing_agent_charts.png")
    plt.show()

