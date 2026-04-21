import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from demand_forecasting   import (model as demand_model,
                                   model_metrics as demand_metrics,
                                   plot_agent_charts as plot_demand_charts)
from inventory_monitoring import (model as inventory_model,
                                   model_metrics as inventory_metrics,
                                   plot_agent_charts as plot_inventory_charts)
from pricing_optimization import (model as pricing_model,
                                   predict_optimal_price,
                                   predict_current_revenue,
                                   model_metrics as pricing_metrics,
                                   plot_agent_charts as plot_pricing_charts)
from ollama_interface import ask_ollama

sns.set_theme(style="whitegrid", palette="muted")

# ── Palette (white background) ───────────────────────────────────────────────
BLUE   = "#2471A3"
TEAL   = "#148F77"
GREEN  = "#1E8449"
RED    = "#C0392B"
ORANGE = "#CA6F1E"
PURPLE = "#7D3C98"
BG     = "white"
PANEL  = "#F4F6F9"

# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases
# ─────────────────────────────────────────────────────────────────────────────
demand_score    = demand_metrics["r2"]
inventory_score = inventory_metrics["accuracy"]
pricing_score   = pricing_metrics["r2"]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: pretty console table
# ─────────────────────────────────────────────────────────────────────────────
def _print_table(df: pd.DataFrame, title: str, color_col: str = None):
    """Print a DataFrame as a clean console table without borders."""
    print(f"\n  {title.upper()}")
    print("  " + "-" * len(title))
    
    # We add indentation to the to_string output so it aligns nicely
    table_str = df.to_string(index=False)
    for line in table_str.split('\n'):
        print(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Fusion chart
# ─────────────────────────────────────────────────────────────────────────────
def _generate_fusion_chart(fusion_df: pd.DataFrame,
                            d_metrics, i_metrics, p_metrics):
    """
    Generate a comprehensive 2×3 Fusion Dashboard combining all 3 agents.
    Includes a unified chart that shows all agents' outputs together.
    """
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        "Retail AI — FUSION RESULT DASHBOARD  (All 3 Agents Combined)",
        fontsize=20, fontweight="bold", color="#1A252F", y=1.01
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    def styled_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="#2C3E50")
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDC3C7")
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1A252F", pad=10)
        ax.xaxis.label.set_color("#2C3E50")
        ax.yaxis.label.set_color("#2C3E50")

    n = len(fusion_df)
    idx = np.arange(n)

    # ── Row 0 ─────────────────────────────────────────────────────────────────

    # [0,0] Demand prediction bar
    ax1 = fig.add_subplot(gs[0, 0])
    bar_c = [BLUE] * n
    bars = ax1.bar(idx, fusion_df["Demand Prediction"], color=bar_c, edgecolor="white")
    ax1.axhline(fusion_df["Demand Prediction"].mean(), color=TEAL, linestyle="--",
                linewidth=1.3, label=f"Mean={fusion_df['Demand Prediction'].mean():.0f}")
    for bar, v in zip(bars, fusion_df["Demand Prediction"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(v)), ha="center", color="#1A252F", fontsize=8, fontweight="bold")
    styled_ax(ax1, f"Demand Prediction per Product\n(R²={d_metrics['r2']})")
    ax1.set_xlabel("Product #"); ax1.set_ylabel("Predicted Qty")
    ax1.set_xticks(idx)
    ax1.legend(fontsize=8)

    # [0,1] Stock risk pie
    ax2 = fig.add_subplot(gs[0, 1])
    rc = fusion_df["Stock Risk"].value_counts()
    colors_pie = [RED if x == "HIGH RISK" else GREEN for x in rc.index]
    wedges, texts, autotexts = ax2.pie(
        rc, labels=rc.index, autopct="%1.1f%%", colors=colors_pie,
        startangle=140,
        explode=[0.08 if x == "HIGH RISK" else 0 for x in rc.index],
        textprops={"color": "#1A252F", "fontsize": 10, "fontweight": "bold"}
    )
    for at in autotexts:
        at.set_color("white")
    ax2.set_facecolor(PANEL)
    ax2.set_title(f"Stock Risk Breakdown\n(Acc={i_metrics['accuracy']} | F1={i_metrics['f1']})",
                  color="#1A252F", fontsize=11, fontweight="bold")

    # [0,2] Current vs Optimal price
    ax3 = fig.add_subplot(gs[0, 2])
    w = 0.35
    ax3.bar(idx - w/2, fusion_df["Pricing Price"],  width=w, color=ORANGE,
            label="Current Price", edgecolor="white")
    ax3.bar(idx + w/2, fusion_df["Optimal Price"],  width=w, color=GREEN,
            label="Optimal Price", edgecolor="white")
    styled_ax(ax3, f"Current vs Optimal Price\n(R²={p_metrics['r2']})")
    ax3.set_xlabel("Product #"); ax3.set_ylabel("Price ($)")
    ax3.set_xticks(idx)
    ax3.legend(fontsize=8)

    # ── Row 1 ─────────────────────────────────────────────────────────────────

    # [1,0] Risk probability scatter colored by risk level
    ax4 = fig.add_subplot(gs[1, 0])
    colors_risk = [RED if r == "HIGH RISK" else GREEN
                   for r in fusion_df["Stock Risk"]]
    scatter = ax4.scatter(idx, fusion_df["Risk Probability"],
                          c=[1 if r == "HIGH RISK" else 0 for r in fusion_df["Stock Risk"]],
                          cmap="RdYlGn_r", edgecolors="white", s=80, linewidths=0.5)
    ax4.axhline(50, color=RED, linestyle="--", linewidth=1, label="50% threshold")
    styled_ax(ax4, f"Risk Probability per Product\n(ROC-AUC={i_metrics['roc_auc']})")
    ax4.set_xlabel("Product #"); ax4.set_ylabel("Risk Prob (%)")
    ax4.set_xticks(idx)
    ax4.legend(fontsize=8)

    # [1,1] Revenue uplift — model-predicted current revenue vs max revenue
    ax5 = fig.add_subplot(gs[1, 1])
    # Use the pricing model to predict volume at CURRENT price (not demand agent)
    est_rev = np.array([
        predict_current_revenue(
            row["Pricing Price"],
            row["Competitor Prices"],
            row["Discounts"],
            row["Elasticity Index"]
        )
        for _, row in fusion_df.iterrows()
    ])
    ax5.plot(idx, est_rev,                         marker="o", color=ORANGE,
             label="Current Revenue (model)", linewidth=1.8)
    ax5.plot(idx, fusion_df["Max Revenue"].values,  marker="s", color=GREEN,
             label="Max Revenue @ Optimal", linewidth=1.8)
    ax5.fill_between(idx, est_rev, fusion_df["Max Revenue"].values,
                     alpha=0.15, color=GREEN)
    styled_ax(ax5, "Revenue Uplift — Model-Based Baseline (Fusion)")
    ax5.set_xlabel("Product #"); ax5.set_ylabel("Revenue ($)")
    ax5.set_xticks(idx)
    ax5.legend(fontsize=8)

    # [1,2] Agent CV scores comparison bar
    ax6 = fig.add_subplot(gs[1, 2])
    agents   = ["Demand\n(R²)", "Inventory\n(Acc)", "Pricing\n(R²)"]
    cv_means = [d_metrics["cv_mean"], i_metrics["cv_mean"], p_metrics["cv_mean"]]
    cv_stds  = [d_metrics["cv_std"],  i_metrics["cv_std"],  p_metrics["cv_std"]]
    bars = ax6.bar(agents, cv_means, yerr=cv_stds,
                   color=[BLUE, GREEN, ORANGE], capsize=8,
                   edgecolor="white", width=0.45)
    ax6.set_ylim(0, 1.15); ax6.set_ylabel("CV Score")
    styled_ax(ax6, "5-Fold CV Comparison\n(All Agents)")
    for bar, m in zip(bars, cv_means):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{m:.3f}", ha="center", color="#1A252F", fontsize=10,
                 fontweight="bold")

    # ── Row 2 – Fusion Summary ────────────────────────────────────────────────

    # [2,0:2] Fusion score radar / spider — replaced with normalised bar chart
    ax7 = fig.add_subplot(gs[2, 0:2])
    metric_names = [
        "Demand R²", "Demand CV",
        "Inv Acc",   "Inv F1",  "Inv ROC",
        "Price R²",  "Price CV"
    ]
    metric_vals = [
        d_metrics["r2"],    d_metrics["cv_mean"],
        i_metrics["accuracy"], i_metrics["f1"], i_metrics["roc_auc"],
        p_metrics["r2"],    p_metrics["cv_mean"]
    ]
    bar_colors = [BLUE, TEAL, GREEN, GREEN, TEAL, ORANGE, RED]
    bars7 = ax7.bar(metric_names, metric_vals, color=bar_colors, edgecolor="white")
    ax7.set_ylim(0, 1.15); ax7.set_ylabel("Score")
    styled_ax(ax7, "Fusion — All Agent Metrics Overview")
    ax7.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars7, metric_vals):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", color="#1A252F", fontsize=9,
                 fontweight="bold")

    # [2,2] Demand vs Max Revenue scatter
    ax8 = fig.add_subplot(gs[2, 2])
    sc = ax8.scatter(
        fusion_df["Demand Prediction"],
        fusion_df["Max Revenue"],
        c=[1 if r == "HIGH RISK" else 0 for r in fusion_df["Stock Risk"]],
        cmap="RdYlGn_r", s=90, edgecolors="white", linewidths=0.5
    )
    for i, row in fusion_df.iterrows():
        ax8.annotate(str(row["Product ID"]),
                     (row["Demand Prediction"], row["Max Revenue"]),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=7, color="#1A252F", fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax8)
    cbar.ax.tick_params(colors="#2C3E50")
    cbar.set_label("Risk (1=High)", color="#2C3E50")
    styled_ax(ax8, "Demand vs Max Revenue\n(colored by Risk)")
    ax8.set_xlabel("Demand Prediction"); ax8.set_ylabel("Max Revenue ($)")

    plt.tight_layout()
    plt.savefig("fusion_result_summary.png", dpi=180, bbox_inches="tight",
                facecolor=BG)
    print("\n  [Saved] fusion_result_summary.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────
def run_agents(
    price=None, promo=None, trend=None,
    stock=None, lead=None, reorder=None,
    comp_price=None, discount=None, elasticity=None,
    num_records=5
):
    use_ui = all(v is not None for v in [
        price, promo, trend,
        stock, lead, reorder,
        comp_price, discount, elasticity
    ])

    # ── MODE 1: STREAMLIT UI ──────────────────────────────────────────────────
    if use_ui:
        promo_val  = 1 if promo == "Yes" else 0
        trend_map  = {"Increasing": 2, "Stable": 1, "Decreasing": 0}
        trend_val  = trend_map[trend]

        df = pd.DataFrame([{
            "Price":                      price,
            "Promotions":                 promo_val,
            "Demand Trend":               trend_val,
            "Stock Levels":               stock,
            "Supplier Lead Time (days)":  lead,
            "Reorder Point":              reorder,
            "Pricing Price":              price,
            "Competitor Prices":          comp_price,
            "Discounts":                  discount,
            "Elasticity Index":           elasticity
        }])

        demand_features         = df[["Price", "Promotions", "Demand Trend"]]
        df["Demand Prediction"] = demand_model.predict(demand_features)

        inv_features         = df[["Stock Levels", "Supplier Lead Time (days)", "Reorder Point"]]
        risk_raw             = inventory_model.predict(inv_features)
        risk_prob            = inventory_model.predict_proba(inv_features)[:, 1]
        df["Stock Risk"]     = ["HIGH RISK" if r else "LOW RISK" for r in risk_raw]
        df["Risk Probability"] = np.round(risk_prob * 100, 1)

        opt_results = df.apply(
            lambda row: predict_optimal_price(
                row["Pricing Price"],
                row["Competitor Prices"],
                row["Discounts"],
                row["Elasticity Index"]
            ), axis=1
        )
        df["Optimal Price"] = [r[0] for r in opt_results]
        df["Max Revenue"]   = [r[1] for r in opt_results]
        df["Revenue Curve"] = [r[2] for r in opt_results]

        result = {
            "Demand":           int(df["Demand Prediction"].iloc[0]),
            "Stock Risk":       df["Stock Risk"].iloc[0],
            "Risk Probability": float(df["Risk Probability"].iloc[0]),
            "Optimal Price":    df["Optimal Price"].iloc[0],
            "Max Revenue":      df["Max Revenue"].iloc[0],
            "Revenue Curve":    df["Revenue Curve"].iloc[0],
            "Demand Metrics":    demand_metrics,
            "Inventory Metrics": inventory_metrics,
            "Pricing Metrics":   pricing_metrics,
        }

        summary = (
            f"Demand Forecast: {result['Demand']} units | "
            f"Stock Risk: {result['Stock Risk']} ({result['Risk Probability']:.1f}%) | "
            f"Optimal Price: ${result['Optimal Price']} | "
            f"Expected Revenue: ${result['Max Revenue']}"
        )
        result["AI Decision"] = ask_ollama(
            f"You are a retail strategy AI. Based on these results provide concise, "
            f"actionable recommendations (3 bullet points max):\n{summary}"
        )
        return result

    # ── MODE 2: CSV BATCH ─────────────────────────────────────────────────────
    else:
        import os
        base_dir = os.path.dirname(__file__)
        demand_df    = pd.read_csv(os.path.join(base_dir, "demand_forecasting.csv"))
        inventory_df = pd.read_csv(os.path.join(base_dir, "inventory_monitoring.csv"))
        pricing_df   = pd.read_csv(os.path.join(base_dir, "pricing_optimization.csv"))

        demand_df["Promotions"]   = demand_df["Promotions"].map({"Yes": 1, "No": 0})
        demand_df["Demand Trend"] = demand_df["Demand Trend"].map(
            {"Increasing": 2, "Stable": 1, "Decreasing": 0}
        )

        df = demand_df.merge(inventory_df, on="Product ID") \
                      .merge(pricing_df,   on="Product ID")
        df = df.drop_duplicates(subset=["Product ID"])
        df = df.rename(columns={"Price_x": "Price", "Price_y": "Pricing Price"})

        # ── Run agents ──────────────────────────────────────────────────────
        demand_features         = df[["Price", "Promotions", "Demand Trend"]]
        df["Demand Prediction"] = demand_model.predict(demand_features)

        inv_features       = df[["Stock Levels", "Supplier Lead Time (days)", "Reorder Point"]]
        risk_raw           = inventory_model.predict(inv_features)
        risk_prob          = inventory_model.predict_proba(inv_features)[:, 1]
        df["Stock Risk"]   = ["HIGH RISK" if r else "LOW RISK" for r in risk_raw]
        df["Risk Probability"] = np.round(risk_prob * 100, 1)

        opt_results = df.apply(
            lambda row: predict_optimal_price(
                row["Pricing Price"],
                row["Competitor Prices"],
                row["Discounts"],
                row["Elasticity Index"]
            ), axis=1
        )
        df["Optimal Price"] = [r[0] for r in opt_results]
        df["Max Revenue"]   = [r[1] for r in opt_results]
        df["Revenue Curve"] = [r[2] for r in opt_results]

        fusion_df = df[
            ["Product ID", "Price", "Pricing Price",
             "Competitor Prices", "Discounts", "Elasticity Index",
             "Demand Prediction", "Stock Risk",
             "Risk Probability", "Optimal Price", "Max Revenue"]
        ].head(num_records).copy()

        # ════════════════════════════════════════════════════════════════════
        # PRINT – DEMAND AGENT
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "═" * 70)
        print("  📊  DEMAND AGENT RESULTS")
        print("═" * 70)
        print(f"  R²: {demand_metrics['r2']}   "
              f"MAE: {demand_metrics['mae']}   "
              f"RMSE: {demand_metrics['rmse']}   "
              f"MAPE: {demand_metrics['mape']}%   "
              f"CV: {demand_metrics['cv_mean']}±{demand_metrics['cv_std']}")
        _print_table(
            fusion_df[["Product ID", "Demand Prediction"]].copy(),
            "📊 Demand Predictions"
        )

        # ── Demand agent charts ─────────────────────────────────────────────
        plot_demand_charts(df["Demand Prediction"].values)

        # ════════════════════════════════════════════════════════════════════
        # PRINT – INVENTORY AGENT
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "═" * 70)
        print("  📦  INVENTORY AGENT RESULTS")
        print("═" * 70)
        print(f"  Acc: {inventory_metrics['accuracy']}   "
              f"Precision: {inventory_metrics['precision']}   "
              f"Recall: {inventory_metrics['recall']}   "
              f"F1: {inventory_metrics['f1']}   "
              f"ROC-AUC: {inventory_metrics['roc_auc']}   "
              f"CV: {inventory_metrics['cv_mean']}±{inventory_metrics['cv_std']}")
        _print_table(
            fusion_df[["Product ID", "Stock Risk", "Risk Probability"]].copy(),
            "📦 Inventory Risk Assessment"
        )

        # ── Inventory agent charts ──────────────────────────────────────────
        plot_inventory_charts(
            list(fusion_df["Stock Risk"]),
            fusion_df["Risk Probability"].values
        )

        # ════════════════════════════════════════════════════════════════════
        # PRINT – PRICING AGENT
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "═" * 70)
        print("  💲  PRICING AGENT RESULTS")
        print("═" * 70)
        print(f"  R²: {pricing_metrics['r2']}   "
              f"MAE: {pricing_metrics['mae']}   "
              f"RMSE: {pricing_metrics['rmse']}   "
              f"MAPE: {pricing_metrics['mape']}%   "
              f"CV: {pricing_metrics['cv_mean']}±{pricing_metrics['cv_std']}")
        _print_table(
            fusion_df[["Product ID", "Pricing Price", "Optimal Price", "Max Revenue"]].copy(),
            "💲 Pricing Optimization"
        )

        # ── Pricing agent charts ────────────────────────────────────────────
        plot_pricing_charts(
            fusion_df["Pricing Price"].values,
            fusion_df["Optimal Price"].values,
            fusion_df["Max Revenue"].values,
            fusion_df["Competitor Prices"].values,
            fusion_df["Discounts"].values,
            fusion_df["Elasticity Index"].values
        )

        # ════════════════════════════════════════════════════════════════════
        # PRINT – FUSION RESULT
        # ════════════════════════════════════════════════════════════════════
        print("  🔗  FUSION RESULT — All 3 Agents Combined")
        

        # Agent performance summary row
        print("\n  ┌─ AGENT PERFORMANCE SUMMARY ──────────────────────────────────────┐")
        print(f"  │  📊 Demand   : R²={demand_metrics['r2']}  "
              f"MAE={demand_metrics['mae']}  CV={demand_metrics['cv_mean']}±{demand_metrics['cv_std']}  │")
        print(f"  │  📦 Inventory: Acc={inventory_metrics['accuracy']}  "
              f"F1={inventory_metrics['f1']}  ROC={inventory_metrics['roc_auc']}  "
              f"CV={inventory_metrics['cv_mean']}±{inventory_metrics['cv_std']}  │")
        print(f"  │  💲 Pricing  : R²={pricing_metrics['r2']}  "
              f"MAE={pricing_metrics['mae']}  CV={pricing_metrics['cv_mean']}±{pricing_metrics['cv_std']}  │")
        print("  └──────────────────────────────────────────────────────────────────┘")

        fusion_display = fusion_df[[
            "Product ID", "Demand Prediction",
            "Stock Risk", "Risk Probability",
            "Optimal Price", "Max Revenue"
        ]].copy()
        fusion_display["Max Revenue"] = fusion_display["Max Revenue"].apply(
            lambda x: f"${x:,.2f}"
        )
        fusion_display["Optimal Price"] = fusion_display["Optimal Price"].apply(
            lambda x: f"${x:.2f}"
        )
        fusion_display["Risk Probability"] = fusion_display["Risk Probability"].apply(
            lambda x: f"{x:.1f}%"
        )
        _print_table(fusion_display, "🔗 FUSION RESULT TABLE")

        # ── AI Strategies ───────────────────────────────────────────────────
        print("\n🤖  Generating AI strategies...")
        strategies = []
        for _, row in fusion_df.iterrows():
            prompt = (
                f"Retail AI – Product {row['Product ID']}:\n"
                f"  Demand: {row['Demand Prediction']} units | "
                f"Stock Risk: {row['Stock Risk']} ({row['Risk Probability']}%) | "
                f"Optimal Price: ${row['Optimal Price']} | Revenue: ${row['Max Revenue']}\n"
                f"Give a brief 2-sentence strategy for this product."
            )
            strategy = ask_ollama(prompt)
            strategies.append(strategy)
            print(f"\n  🏷  Product {row['Product ID']}: {strategy}")

        fusion_df["AI Strategy"] = strategies
        fusion_df.to_csv("agent_collaboration_output.csv", index=False)

        # ── Fusion chart ────────────────────────────────────────────────────
        _generate_fusion_chart(fusion_df, demand_metrics, inventory_metrics, pricing_metrics)

        
        print("  ✅  DONE — Results saved to agent_collaboration_output.csv")
        

        return fusion_df
