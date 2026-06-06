import os
import streamlit as st

# ── Groq client (lazy init) ───────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from groq import Groq
        # Try Streamlit secrets first (cloud deploy), then env var (local)
        api_key = None
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY", "")

        if not api_key:
            return None
        _client = Groq(api_key=api_key)
        return _client
    except Exception:
        return None


def ask_ollama(prompt: str) -> str:
    """
    Drop-in replacement for the original ask_ollama function.
    Now calls Groq (llama3-8b-8192) instead of a local Ollama instance.
    Falls back to a rule-based response when the API key is not configured.
    """
    client = _get_client()
    if client is None:
        return _rule_based_strategy(prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retail supply chain strategy AI. "
                        "Give concise, actionable recommendations based on the data provided. "
                        "Use bullet points. Be specific and quantitative where possible."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=400,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return _rule_based_strategy(prompt)


def _rule_based_strategy(prompt: str) -> str:
    """
    Generates a smart, data-driven strategy text from the prompt itself
    without requiring any external API. Used as fallback.
    """
    import re

    # Extract key values from the prompt using regex
    demand_match   = re.search(r"Demand[:\s]+(\d+)", prompt, re.IGNORECASE)
    risk_match     = re.search(r"Stock Risk[:\s]+(HIGH RISK|LOW RISK)", prompt, re.IGNORECASE)
    risk_pct_match = re.search(r"\((\d+\.?\d*)%\)", prompt)
    price_match    = re.search(r"Optimal Price[:\s]+\$?([\d.]+)", prompt, re.IGNORECASE)
    revenue_match  = re.search(r"(?:Revenue|Max Revenue)[:\s]+\$?([\d,]+)", prompt, re.IGNORECASE)

    demand  = int(demand_match.group(1))          if demand_match   else None
    risk    = risk_match.group(1).upper()          if risk_match     else "UNKNOWN"
    risk_pct= float(risk_pct_match.group(1))       if risk_pct_match else None
    price   = float(price_match.group(1))          if price_match    else None
    revenue = revenue_match.group(1).replace(",","") if revenue_match else None

    lines = []

    # Demand recommendation
    if demand is not None:
        if demand > 200:
            lines.append(f"📈 **Demand is strong at {demand} units** — consider increasing stock levels and pre-positioning inventory closer to distribution hubs to avoid lost sales.")
        elif demand < 80:
            lines.append(f"📉 **Demand is low at {demand} units** — reduce procurement orders, run a targeted promotion to clear existing stock, and avoid overordering.")
        else:
            lines.append(f"📊 **Demand is moderate at {demand} units** — maintain current stock with minor adjustments; monitor weekly trends before committing to bulk orders.")

    # Inventory / risk recommendation
    if risk == "HIGH RISK":
        pct_str = f" ({risk_pct:.1f}% stockout probability)" if risk_pct else ""
        lines.append(f"⚠️ **Stockout risk is HIGH{pct_str}** — trigger reorder immediately, consider safety-stock buffers of 20–30% above the reorder point, and negotiate expedited lead times with suppliers.")
    elif risk == "LOW RISK":
        pct_str = f" ({risk_pct:.1f}% stockout probability)" if risk_pct else ""
        lines.append(f"✅ **Inventory risk is LOW{pct_str}** — stock levels are healthy. Delay replenishment to the next scheduled cycle to avoid holding costs and tie-up of working capital.")

    # Pricing recommendation
    if price is not None:
        rev_str = f" estimated to generate ${int(revenue):,}" if revenue and revenue.isdigit() else ""
        lines.append(f"💲 **Set price to ${price:.2f}**{rev_str} — this is the revenue-maximising point identified by the pricing model. A/B test this price for 7–14 days before full rollout.")

    if not lines:
        lines.append("⚙️ No specific recommendations could be generated from the available data. Please verify the input parameters and re-run the optimization engine.")

    return "\n\n".join(lines)
