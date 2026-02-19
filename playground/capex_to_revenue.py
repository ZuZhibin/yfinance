"""
CAPEX / Free Cash Flow ratio for GOOGL, META, AMZN, MSFT

Uses yfinance to fetch annual cash flow statements,
then calculates Capital Expenditure as a percentage of Free Cash Flow.

Note: Yahoo Finance API only provides ~4-5 years of annual data,
so the actual range depends on what Yahoo returns.
"""

import sys
sys.path.insert(0, "..")

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────
TICKERS = {
    "GOOGL": "Google",
    "META":  "Meta",
    "AMZN":  "Amazon",
    "MSFT":  "Microsoft",
}

# ── Fetch data ─────────────────────────────────────────────────
records = []

for symbol, name in TICKERS.items():
    print(f"Fetching {name} ({symbol}) ...")
    tk = yf.Ticker(symbol)

    cashflow = tk.get_cash_flow(pretty=False)

    for col in sorted(cashflow.columns):
        year = col.year
        if year > 2025:
            continue

        capex = cashflow.at["CapitalExpenditure", col] if "CapitalExpenditure" in cashflow.index else None
        fcf = cashflow.at["FreeCashFlow", col] if "FreeCashFlow" in cashflow.index else None

        if pd.notna(capex) and pd.notna(fcf) and float(fcf) != 0:
            capex_abs = abs(float(capex))
            fcf_val = float(fcf)
            ratio = capex_abs / abs(fcf_val) * 100
            records.append({
                "Company": name,
                "FY Ending": str(col.date()),
                "Year": year,
                "CAPEX ($B)": round(capex_abs / 1e9, 1),
                "FCF ($B)": round(fcf_val / 1e9, 1),
                "CAPEX/FCF (%)": round(ratio, 2),
            })

# ── Build DataFrame ────────────────────────────────────────────
df = pd.DataFrame(records).sort_values(["Company", "Year"])

print("\n===== Raw Data =====\n")
print(df[["Company", "FY Ending", "CAPEX ($B)", "FCF ($B)", "CAPEX/FCF (%)"]].to_string(index=False))

pivot = df.pivot(index="Year", columns="Company", values="CAPEX/FCF (%)")
col_order = [c for c in ["Google", "Meta", "Amazon", "Microsoft"] if c in pivot.columns]
pivot = pivot[col_order]

print("\n===== CAPEX / FCF (%) =====\n")
print(pivot.to_string())
print()

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "Google":    "#4285F4",
    "Meta":      "#0668E1",
    "Amazon":    "#FF9900",
    "Microsoft": "#7FBA00",
}

for company in pivot.columns:
    series = pivot[company].dropna()
    ax.plot(
        series.index, series.values,
        marker="o", linewidth=2.5, markersize=7,
        label=company, color=colors[company],
    )
    for x, y in zip(series.index, series.values):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

years = sorted(df["Year"].unique())
ax.set_title("CAPEX / Free Cash Flow — Big Tech Comparison", fontsize=16, fontweight="bold")
ax.set_xlabel("Fiscal Year", fontsize=12)
ax.set_ylabel("CAPEX / FCF (%)", fontsize=12)
ax.set_xticks(years)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
output_path = "capex_to_fcf.png"
plt.savefig(output_path, dpi=150)
print(f"Chart saved to playground/{output_path}")
