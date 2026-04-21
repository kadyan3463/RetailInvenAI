import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

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


import os

def load_data():
    base_dir = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(base_dir, "inventory_monitoring.csv"))


def train_model():
    df = load_data()

    X = df[["Stock Levels", "Supplier Lead Time (days)", "Reorder Point"]]
    y = (df["Stockout Frequency"] > 5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]

    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_prob)
    cm         = confusion_matrix(y_test, y_pred)

    cv_scores  = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    cv_mean    = cv_scores.mean()
    cv_std     = cv_scores.std()

    feat_imp = dict(zip(X.columns, model.feature_importances_))

    metrics = {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "roc_auc":   round(roc_auc, 4),
        "cv_mean":   round(cv_mean, 4),
        "cv_std":    round(cv_std, 4),
        "feat_imp":  {k: round(v, 4) for k, v in feat_imp.items()},
        # store for plotting
        "_y_test":   y_test.values,
        "_y_pred":   y_pred,
        "_y_prob":   y_prob,
        "_cm":       cm,
        "_X":        X,
        "_y":        y,
    }

    return model, metrics


# Train model once at import
model, model_metrics = train_model()
model_score = model_metrics["accuracy"]


def predict_stock_risk(stock, lead_time, reorder):
    input_data = pd.DataFrame(
        [[stock, lead_time, reorder]],
        columns=[
            "Stock Levels",
            "Supplier Lead Time (days)",
            "Reorder Point"
        ]
    )

    risk      = model.predict(input_data)[0]
    risk_prob = model.predict_proba(input_data)[0][1]

    label = "HIGH RISK" if risk else "LOW RISK"
    return label, round(float(risk_prob), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent chart function (called during CSV batch execution)
# ─────────────────────────────────────────────────────────────────────────────
def plot_agent_charts(risk_labels: list, risk_probs: np.ndarray):
    """
    Display 4 inventory-agent charts during CSV execution.

    Parameters
    ----------
    risk_labels : list of str
        "HIGH RISK" / "LOW RISK" labels for all processed products.
    risk_probs  : array-like
        Risk probability values (0–100) for all processed products.
    """
    print("\n" + "─" * 70)
    print("  📦  INVENTORY AGENT — Charts")
    print("─" * 70)

    y_test = model_metrics["_y_test"]
    y_pred = model_metrics["_y_pred"]
    y_prob = model_metrics["_y_prob"]
    cm     = model_metrics["_cm"]
    X      = model_metrics["_X"]
    y      = model_metrics["_y"]

    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    fig.suptitle(
        "Inventory Monitoring Agent  —  Analysis Dashboard",
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

    # ── 1. Risk breakdown pie ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    from collections import Counter
    cnt = Counter(risk_labels)
    labels_pie = list(cnt.keys())
    vals_pie   = list(cnt.values())
    colors_pie = [RED if l == "HIGH RISK" else GREEN for l in labels_pie]
    wedges, texts, autotexts = ax1.pie(
        vals_pie, labels=labels_pie, autopct="%1.1f%%",
        colors=colors_pie, startangle=140,
        explode=[0.08 if l == "HIGH RISK" else 0 for l in labels_pie],
        textprops={"color": "#1A252F", "fontsize": 10, "fontweight": "bold"}
    )
    for at in autotexts:
        at.set_color("white")
    ax1.set_facecolor(PANEL)
    ax1.set_title(f"Stock Risk Breakdown\n(n={len(risk_labels)} products)",
                  color="#1A252F", fontsize=11, fontweight="bold")

    # ── 2. Confusion matrix heatmap ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"], ax=ax2,
                linewidths=0.5, linecolor="#BDC3C7",
                cbar_kws={"shrink": 0.8})
    styled_ax(ax2, f"Confusion Matrix\nAcc={model_metrics['accuracy']} | F1={model_metrics['f1']}")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.tick_params(colors="#2C3E50")


    # ── 3. Feature importance horizontal bar ──────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    fi     = model_metrics["feat_imp"]
    feats  = list(fi.keys())
    vals   = list(fi.values())
    colors_fi = [BLUE, TEAL, PURPLE]
    bars = ax4.barh(feats, vals, color=colors_fi, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax4.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", color="#1A252F", fontsize=9, fontweight="bold")
    styled_ax(ax4, "Feature Importances")
    ax4.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("inventory_agent_charts.png", dpi=180, bbox_inches="tight",
                facecolor=BG)
    print("  [Saved] inventory_agent_charts.png")
    plt.show()

    
