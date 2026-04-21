import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from agent_collaboration import run_agents
from pricing_optimization import predict_sales

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailInvenAI | Intelligence Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #334155; }

/* Main app background */
.stApp {
    background: #f8fafc;
    background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
    background-size: 24px 24px;
}

/* Agent header banners */
.agent-header {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 1.15rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.demand-header   { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
.inventory-header{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
.pricing-header  { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }
.fusion-header   { background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); }

/* Accuracy table */
.acc-table { width:100%; border-collapse: separate; border-spacing: 0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-top:8px; }
.acc-table th {
    background: #f1f5f9;
    color: #475569;
    padding: 10px 14px;
    text-align:left;
    font-size:0.9rem;
    font-weight: 600;
}
.acc-table td {
    background: #ffffff;
    padding: 10px 14px;
    color: #334155;
    font-size: 0.9rem;
    border-top: 1px solid #f1f5f9;
}

/* Info box */
.ai-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-left: 5px solid #8b5cf6;
    border-radius: 12px;
    padding: 20px 24px;
    color: #334155;
    line-height: 1.8;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    font-size: 1.05rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #f1f5f9;
    box-shadow: 2px 0 12px rgba(0,0,0,0.03);
}

/* Reduce sidebar top padding */
[data-testid="stSidebarUserContent"] {
    padding-top: 2rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    margin-top: -1.5rem !important;
    padding-top: 0 !important;
}

/* Clean, decent color for Product Configuration */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #f8fafc;
    border-radius: 10px 10px 0 0;
    color: #334155 !important;
    font-weight: 600;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 14px;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(79, 70, 229, 0.4);
}

/* KPIs Styling */
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    color: #1e293b !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 1.05rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; padding: 2.5rem 0 1.5rem 0; margin-top: -50px;'>
    <div style='
        display: inline-block;
        padding: 6px 18px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #64748b;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    '>
        🚀 AI-Powered Operations
    </div >
    <div style='font-size: 4.2rem; font-weight: 700; letter-spacing: -2px; line-height: 1.1; color: #0f172a; margin-bottom: 1rem;'>
        🚀<span style='color: #3b82f6;'>RetailInvenAI </span>Dashboard
    </div>
    <div style='font-size: 1.25rem; font-weight: 300; color: #475569; max-width: 600px; margin: 0 auto; letter-spacing: 0.2px;'>
        Advanced Multi-Agent Supply Chain Optimization <span style='margin: 0 10px; color: #cbd5e1;'>|</span> Real-time Insights
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Product Configuration")

    with st.expander("📦 Product & Stock", expanded=True):
        price   = st.number_input("Current Price ($)",  1.0,  500.0,  40.0, step=0.5)
        stock   = st.number_input("Stock Level (units)", 0,   5000,   120)
        lead    = st.number_input("Lead Time (days)",    1,     90,     7)
        reorder = st.number_input("Reorder Point",       0,   1000,    50)

    with st.expander("📈 Market Context", expanded=True):
        promo      = st.selectbox("Active Promotion",  ["Yes", "No"])
        trend      = st.selectbox("Demand Trend",      ["Increasing", "Stable", "Decreasing"])
        comp_price = st.number_input("Competitor Price ($)", 1.0, 500.0, 42.0, step=0.5)
        discount   = st.number_input("Discount %",    0.0, 100.0, 10.0, step=0.5)
        elasticity = st.number_input("Elasticity Index", 0.1, 5.0, 1.4, step=0.05)

    st.divider()
    run_btn = st.button("▶ RUN OPTIMIZATION ENGINE", use_container_width=True)

# ── Welcome screen ────────────────────────────────────────────────────────────
if not run_btn:
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 1px solid #bae6fd; border-radius: 16px; padding: 24px; color: #0369a1; box-shadow: 0 4px 10px rgba(0,0,0,0.03); height: 100%;'>
    <h3 style='color: #0284c7; margin-top: 0; display: flex; align-items: center; gap: 8px;'>📈 Demand Agent</h3>
    <p style='margin-bottom: 0; line-height: 1.6; font-size: 1.05rem;'>XGBoost-powered demand forecasting with cross-validated R² accuracy, MAE and MAPE reporting.</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #bbf7d0; border-radius: 16px; padding: 24px; color: #166534; box-shadow: 0 4px 10px rgba(0,0,0,0.03); height: 100%;'>
    <h3 style='color: #15803d; margin-top: 0; display: flex; align-items: center; gap: 8px;'>📦 Inventory Agent</h3>
    <p style='margin-bottom: 0; line-height: 1.6; font-size: 1.05rem;'>Random Forest classifier with F1, ROC-AUC and risk probability. Balanced for class imbalance.</p>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div style='background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); border: 1px solid #fed7aa; border-radius: 16px; padding: 24px; color: #9a3412; box-shadow: 0 4px 10px rgba(0,0,0,0.03); height: 100%;'>
    <h3 style='color: #c2410c; margin-top: 0; display: flex; align-items: center; gap: 8px;'>💲 Pricing Agent</h3>
    <p style='margin-bottom: 0; line-height: 1.6; font-size: 1.05rem;'>Revenue-maximising price search over ±25% range. Returns full revenue curve for analysis.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Welcome!** Please configure the product parameters in the sidebar and click **RUN OPTIMIZATION ENGINE** to begin.", icon="👋")
    st.stop()

# ── Run agents ────────────────────────────────────────────────────────────────
with st.spinner("🤖 Multi-Agent system computing optimal strategy…"):
    R = run_agents(
        price, promo, trend,
        stock, lead, reorder,
        comp_price, discount, elasticity
    )

dm = R["Demand Metrics"]
im = R["Inventory Metrics"]
pm = R["Pricing Metrics"]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – TOP KPI STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

k1.metric("🔮 Predicted Demand",   f"{R['Demand']:,} units",
          delta=f"R² {dm['r2']}")
k2.metric("⚠️ Stock Risk",         R["Stock Risk"],
          delta=f"{R['Risk Probability']:.1f}% probability",
          delta_color="inverse" if R["Stock Risk"] == "HIGH RISK" else "normal")
k3.metric("💲 Optimal Price",       f"${R['Optimal Price']}",
          delta=f"Δ ${round(R['Optimal Price'] - price, 2)}")
k4.metric("💰 Max Revenue",         f"${R['Max Revenue']:,.0f}",
          delta=f"R² {pm['r2']}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – AGENT ACCURACY PANELS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Agent Accuracy Metrics")
a1, a2, a3 = st.columns(3)

# ── Demand agent ──────────────────────────────────────────────────────────────
with a1:
    st.markdown("""<div class="agent-header demand-header">📈 Demand Forecasting Agent<br>
    <span style="font-weight:300;font-size:0.85rem;">XGBoost Regressor · 200 Trees</span></div>""",
                unsafe_allow_html=True)
    st.markdown(f"""
    <table class="acc-table">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>R² Score</td><td><b>{dm['r2']}</b></td></tr>
    <tr><td>MAE</td><td>{dm['mae']}</td></tr>
    <tr><td>RMSE</td><td>{dm['rmse']}</td></tr>
    <tr><td>MAPE</td><td>{dm['mape']}%</td></tr>
    <tr><td>CV R² (5-fold)</td><td>{dm['cv_mean']} ± {dm['cv_std']}</td></tr>
    </table>""", unsafe_allow_html=True)

# ── Inventory agent ───────────────────────────────────────────────────────────
with a2:
    st.markdown("""<div class="agent-header inventory-header">📦 Inventory Monitoring Agent<br>
    <span style="font-weight:300;font-size:0.85rem;">Random Forest · 200 Trees · Balanced</span></div>""",
                unsafe_allow_html=True)
    st.markdown(f"""
    <table class="acc-table">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Accuracy</td><td><b>{im['accuracy']}</b></td></tr>
    <tr><td>Precision</td><td>{im['precision']}</td></tr>
    <tr><td>Recall</td><td>{im['recall']}</td></tr>
    <tr><td>F1 Score</td><td>{im['f1']}</td></tr>
    <tr><td>ROC-AUC</td><td>{im['roc_auc']}</td></tr>
    <tr><td>CV Acc (5-fold)</td><td>{im['cv_mean']} ± {im['cv_std']}</td></tr>
    </table>""", unsafe_allow_html=True)

# ── Pricing agent ─────────────────────────────────────────────────────────────
with a3:
    st.markdown("""<div class="agent-header pricing-header">💲 Pricing Optimization Agent<br>
    <span style="font-weight:300;font-size:0.85rem;">XGBoost Regressor · 200 Trees</span></div>""",
                unsafe_allow_html=True)
    st.markdown(f"""
    <table class="acc-table">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>R² Score</td><td><b>{pm['r2']}</b></td></tr>
    <tr><td>MAE</td><td>{pm['mae']}</td></tr>
    <tr><td>RMSE</td><td>{pm['rmse']}</td></tr>
    <tr><td>MAPE</td><td>{pm['mape']}%</td></tr>
    <tr><td>CV R² (5-fold)</td><td>{pm['cv_mean']} ± {pm['cv_std']}</td></tr>
    </table>""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📉 Analytical Visualizations")

row1_c1, row1_c2 = st.columns(2)

# ── Chart 1: Revenue Curve ────────────────────────────────────────────────────
with row1_c1:
    st.markdown("#### 💰 Revenue Sensitivity Curve")
    curve = R["Revenue Curve"]                    # list of (price, vol, revenue)
    prices_c  = [c[0] for c in curve]
    revenues  = [c[2] for c in curve]

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=prices_c, y=revenues, mode="lines+markers",
        line=dict(color="#8e54e9", width=3),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(142,84,233,0.12)",
        name="Revenue"
    ))
    fig_rev.add_vline(x=R["Optimal Price"], line_dash="dash", line_color="#2ecc71",
                      annotation_text=f"Optimal ${R['Optimal Price']}", annotation_font_color="#2ecc71")
    fig_rev.add_vline(x=price, line_dash="dot", line_color="#f39c12",
                      annotation_text=f"Current ${price}", annotation_font_color="#f39c12")
    fig_rev.update_layout(
        template="plotly_white", margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Price ($)", yaxis_title="Est. Revenue ($)",
        height=350, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_rev, use_container_width=True)

# ── Chart 2: Stockout Risk Gauge ──────────────────────────────────────────────
with row1_c2:
    st.markdown("#### ⚠️ Stockout Risk Gauge")
    risk_val = round(R["Risk Probability"], 1)
    gauge_color = "#e74c3c" if risk_val > 50 else "#2ecc71"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_val,
        delta={"reference": 50, "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#2ecc71"}},
        title={"text": f"Stock Risk Probability<br><span style='font-size:0.9em;color:#aaa'>{R['Stock Risk']}</span>",
               "font": {"size": 16}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar":   {"color": gauge_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  35], "color": "rgba(46,204,113,0.25)"},
                {"range": [35, 65], "color": "rgba(243,156,18,0.25)"},
                {"range": [65,100], "color": "rgba(231,76,60,0.25)"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8, "value": risk_val
            }
        },
        number={"suffix": "%", "font": {"size": 36, "color": gauge_color}}
    ))
    fig_gauge.update_layout(
        margin=dict(l=30, r=30, t=60, b=20), height=350,
        paper_bgcolor="rgba(0,0,0,0)", font_color="#374151"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── Row 2 ──────────────────────────────────────────────────────────────────────
row2_c1, row2_c2 = st.columns(2)

# ── Chart 3: Feature Importance ───────────────────────────────────────────────
with row2_c1:
    st.markdown("#### 🔍 Pricing Model – Feature Importance")
    fi = pm["feat_imp"]
    fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}) \
              .sort_values("Importance", ascending=True)
    colors_fi = ["#4776e6", "#8e54e9", "#f7971e", "#ffd200"]
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Feature", color_discrete_sequence=colors_fi,
                    template="plotly_white")
    fig_fi.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), height=300, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Chart 4: Inventory feature importance ────────────────────────────────────
with row2_c2:
    st.markdown("#### 🔍 Inventory Model – Feature Importance")
    fi_inv = im["feat_imp"]
    fi_inv_df = pd.DataFrame({"Feature": list(fi_inv.keys()), "Importance": list(fi_inv.values())}) \
                  .sort_values("Importance", ascending=True)
    colors_inv = ["#11998e", "#38ef7d", "#3498db"]
    fig_inv_fi = px.bar(fi_inv_df, x="Importance", y="Feature", orientation="h",
                        color="Feature", color_discrete_sequence=colors_inv,
                        template="plotly_white")
    fig_inv_fi.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), height=300, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_inv_fi, use_container_width=True)

# ── Chart 5: Cross-validation radar / bar ────────────────────────────────────
st.markdown("#### 📐 Cross-Validation Performance (All Agents)")
cv_df = pd.DataFrame({
    "Agent":    ["Demand (R²)", "Inventory (Acc)", "Pricing (R²)"],
    "CV Mean":  [dm["cv_mean"], im["cv_mean"], pm["cv_mean"]],
    "CV Std":   [dm["cv_std"],  im["cv_std"],  pm["cv_std"]],
})
fig_cv = go.Figure()
fig_cv.add_trace(go.Bar(
    x=cv_df["Agent"], y=cv_df["CV Mean"],
    error_y=dict(type="data", array=cv_df["CV Std"].tolist(), visible=True, color="#1f2937"),
    marker_color=["#3b82f6", "#22c55e", "#f97316"],
    marker_line_color="#1f2937", marker_line_width=1,
    text=[f"{v:.3f}" for v in cv_df["CV Mean"]], textposition="outside",
    name="CV Score"
))
fig_cv.update_layout(
    template="plotly_white", yaxis=dict(range=[0, 1.15], title="Score"),
    margin=dict(l=20, r=20, t=20, b=20), height=300,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False
)
st.plotly_chart(fig_cv, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – AI DECISION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🤖 Strategic Intelligence Report")

# Clean up AI output formatting to strict HTML
ai_raw = str(R.get("AI Decision", ""))
# 1. Fix numbers separated by newlines (e.g., '2.\nMonitor' -> '2. Monitor')
ai_raw = re.sub(r'(\d+\.)\s*\n\s*', r'\1 ', ai_raw)

# 1.5. Force bullet points to start on a new paragraph if the AI merged them
# Matches a sentence end (.!?) followed by spaces and a list number
ai_raw = re.sub(r'([.!?])\s+(?=\d+\.\s)', r'\1\n\n', ai_raw)
# Matches a single newline before a list number
ai_raw = re.sub(r'(?<!\n)\n(?=\d+\.\s)', '\n\n', ai_raw)

# 2. Split into distinct paragraphs by 2 or more newlines
paragraphs = [p.strip() for p in re.split(r'\n{2,}|\r\n\r\n', ai_raw) if p.strip()]

# 3. Clean each paragraph (remove internal newlines) and join with standard HTML break
clean_paragraphs = [p.replace('\n', ' ').replace('\r', '') for p in paragraphs]
final_html = "<br><br>".join(clean_paragraphs)

# 4. Handle any basic bold markdown manually
final_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', final_html)

st.markdown(f"""<div class="ai-box">{final_html}</div>""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – STRUCTURED AGENT OUTPUT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Structured Agent Output Summary")

summary_df = pd.DataFrame([{
    "Parameter":   "Current Price ($)",     "Value": f"${price}",                          "Agent": "Input"}])
rows = [
    ("Current Price ($)",         f"${price}",                                  "Input"),
    ("Promotion Active",          promo,                                         "Input"),
    ("Demand Trend",              trend,                                          "Input"),
    ("Competitor Price ($)",      f"${comp_price}",                              "Input"),
    ("Discount (%)",              f"{discount}%",                                "Input"),
    ("Elasticity Index",          str(elasticity),                               "Input"),
    ("Stock Level",               f"{stock} units",                              "Input"),
    ("Lead Time",                 f"{lead} days",                                "Input"),
    ("Reorder Point",             str(reorder),                                  "Input"),
    ("Predicted Demand",          f"{R['Demand']:,} units",                     "📈 Demand Agent"),
    ("Demand R²",                 str(dm["r2"]),                                 "📈 Demand Agent"),
    ("Demand MAE",                str(dm["mae"]),                                "📈 Demand Agent"),
    ("Demand RMSE",               str(dm["rmse"]),                               "📈 Demand Agent"),
    ("Demand MAPE",               f"{dm['mape']}%",                             "📈 Demand Agent"),
    ("Stock Risk Level",          R["Stock Risk"],                               "📦 Inventory Agent"),
    ("Risk Probability",          f"{R['Risk Probability']:.1f}%",              "📦 Inventory Agent"),
    ("Inventory Accuracy",        str(im["accuracy"]),                           "📦 Inventory Agent"),
    ("Inventory F1 Score",        str(im["f1"]),                                 "📦 Inventory Agent"),
    ("Inventory ROC-AUC",         str(im["roc_auc"]),                            "📦 Inventory Agent"),
    ("Optimal Price ($)",         f"${R['Optimal Price']}",                     "💲 Pricing Agent"),
    ("Price Change",              f"${round(R['Optimal Price'] - price, 2)}",   "💲 Pricing Agent"),
    ("Max Revenue ($)",           f"${R['Max Revenue']:,.0f}",                  "💲 Pricing Agent"),
    ("Pricing R²",                str(pm["r2"]),                                 "💲 Pricing Agent"),
    ("Pricing MAPE",              f"{pm['mape']}%",                             "💲 Pricing Agent"),
]
out_df = pd.DataFrame(rows, columns=["Parameter", "Value", "Agent"])

def color_agent(val):
    colors = {
        "📈 Demand Agent":   "background-color:#e0f2fe;color:#0369a1;font-weight:500;",
        "📦 Inventory Agent":"background-color:#dcfce7;color:#15803d;font-weight:500;",
        "💲 Pricing Agent":  "background-color:#ffedd5;color:#c2410c;font-weight:500;",
        "Input":             "background-color:#f1f5f9;color:#475569;",
    }
    return colors.get(val, "")

styled = out_df.style.applymap(color_agent, subset=["Agent"]) \
               .set_properties(**{"font-size": "0.95rem", "border-bottom": "1px solid #f1f5f9"})
st.dataframe(styled, use_container_width=True, hide_index=True)