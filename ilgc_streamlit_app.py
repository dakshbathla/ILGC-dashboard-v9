# =============================================================================
# ILGC STREAMLIT DASHBOARD
# Run: streamlit run ilgc_streamlit_app.py
# Install: pip install streamlit plotly pandas yfinance statsmodels
#
# Place this file in the same directory as ilgc_dcf_nifty50_final.py
# After running the pipeline, it produces ilgc_full_dashboard_data.json
# This app reads that JSON and displays everything.
#
# If no JSON exists yet, the app shows a "Run Pipeline" button that
# executes the pipeline inline and then refreshes.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ILGC — Equity Valuation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Bloomberg-dark theme ─────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #060c18; color: #d1d5db; }
[data-testid="stSidebar"] { background-color: #080f1e; border-right: 1px solid #1f2937; }
[data-testid="stSidebar"] * { color: #9ca3af !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0d1321;
    border: 0.5px solid #1f2937;
    border-radius: 8px;
    padding: 12px;
}
[data-testid="stMetricLabel"] > div { color: #6b7280 !important; font-size: 11px; }
[data-testid="stMetricValue"] > div { color: #e5e7eb !important; font-size: 20px; font-weight: 700; font-family: 'Courier New', monospace; }
[data-testid="stMetricDelta"] > div { font-size: 12px; font-family: 'Courier New', monospace; }

/* DataFrames */
[data-testid="stDataFrame"] { background: #080f1e; }
.stDataFrame { font-family: 'Courier New', monospace; font-size: 12px; }

/* Tabs */
.stTabs [data-baseweb="tab"] { background: #0d1321; color: #6b7280; font-size: 12px; font-family: 'Courier New', monospace; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #3b82f6; border-bottom: 2px solid #3b82f6; }

/* Headers */
h1 { color: #3b82f6 !important; font-family: 'Courier New', monospace !important; letter-spacing: 0.1em; font-size: 22px !important; }
h2, h3 { color: #6b7280 !important; font-family: 'Courier New', monospace !important; font-size: 13px !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* Selectbox / inputs */
.stSelectbox > div > div { background: #0d1321; color: #d1d5db; border: 0.5px solid #1f2937; }
.stTextInput > div > div > input { background: #0d1321; color: #d1d5db; border: 0.5px solid #1f2937; font-family: 'Courier New', monospace; }

/* Divider */
hr { border-color: #1f2937; }

/* Signal badges */
.badge-buy  { background: rgba(34,197,94,0.15);  color: #22c55e; padding: 2px 10px; border-radius: 4px; font-weight: 700; font-size: 11px; }
.badge-sell { background: rgba(239,68,68,0.15);  color: #ef4444; padding: 2px 10px; border-radius: 4px; font-weight: 700; font-size: 11px; }
.badge-hold { background: rgba(245,158,11,0.15); color: #f59e0b; padding: 2px 10px; border-radius: 4px; font-weight: 700; font-size: 11px; }
.badge-na   { background: rgba(107,114,128,0.15);color: #6b7280; padding: 2px 10px; border-radius: 4px; font-weight: 700; font-size: 11px; }

/* Section label */
.section-label {
    font-size: 9px; font-weight: 700; color: #374151;
    letter-spacing: 0.12em; text-transform: uppercase;
    border-bottom: 0.5px solid #1f2937;
    padding-bottom: 4px; margin-bottom: 8px; margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_FILE = "ilgc_full_dashboard_data.json"
SIGNAL_COLORS = {"BUY": "#22c55e", "HOLD": "#f59e0b", "SELL": "#ef4444", "N/A": "#6b7280"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#060c18", plot_bgcolor="#060c18",
    font=dict(family="Courier New, monospace", color="#6b7280", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
)

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    with open(DATA_FILE) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    # Ensure numeric columns
    num_cols = ["Price", "IntrinsicPrice_DCF", "CompsPrice", "Triangulated_Price",
                "Triangulated_Upside", "WACC", "Ke", "Kd_AfterTax", "Beta",
                "BlendedGrowth", "FundamentalGrowth", "ImpliedGrowth",
                "PE_Ratio", "ROE", "ROA", "ProfitMargin", "EPS_TTM", "DPS",
                "BookValue", "EBITDA_Cr", "EBIT_Cr", "Revenue_Cr",
                "MarketCap_Cr", "TotalDebt_Cr", "TotalCash_Cr", "NetDebt_EBITDA",
                "PV_ExplicitFCFF_Cr", "PV_Terminal_Cr", "EV_Cr",
                "ICR", "SyntheticSpread", "We", "Wd", "NOPAT_Cr"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ── Helpers ────────────────────────────────────────────────────────────────────
def pct(v, digits=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{'+' if v >= 0 else ''}{v*100:.{digits}f}%"

def inr(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"₹{v:,.0f}"

def inr_cr(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if abs(v) >= 1e5: return f"₹{v/1e5:.1f}L Cr"
    return f"₹{v:,.0f} Cr"

def signal_badge(sig):
    cls = {"BUY":"badge-buy","SELL":"badge-sell","HOLD":"badge-hold","N/A":"badge-na"}.get(sig,"badge-na")
    return f'<span class="{cls}">{sig}</span>'

def sensitivity_grid(wacc, growth, nopat_cr, total_debt_cr, total_cash_cr, shares):
    """5×5 WACC × terminal growth sensitivity grid. Returns styled DataFrame."""
    wacc_range   = [wacc + d for d in [-0.02, -0.01, 0, +0.01, +0.02]]
    growth_range = [max(0.02, growth + d) for d in [-0.01, -0.005, 0, +0.005, +0.01]]

    rows = {}
    for w in wacc_range:
        row = {}
        for g in growth_range:
            col_label = f"g={g*100:.1f}%"
            if w <= g:
                row[col_label] = np.nan
                continue
            tv_cr = (nopat_cr * (1 + g)) / (w - g)
            eq_cr = tv_cr - total_debt_cr + total_cash_cr
            px    = (eq_cr * 1e7) / shares if shares > 0 else np.nan
            row[col_label] = round(max(0, px), 0)
        rows[f"WACC={w*100:.1f}%"] = row

    df = pd.DataFrame(rows).T
    return df


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def sidebar(df):
    st.sidebar.markdown("### ILGC v9")
    st.sidebar.markdown("Nifty 50 · DCF + Comps + Triangulation")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Parameters**")
    st.sidebar.markdown(f"`Rf` 4.80% (G-sec − default spread)")
    st.sidebar.markdown(f"`ERP` 7.46% (Damodaran Jul 2025)")
    st.sidebar.markdown(f"`Rm` 12.26%")
    st.sidebar.markdown("---")

    # Filters
    st.sidebar.markdown("**Filters**")
    signals = ["ALL"] + sorted(df["Signal"].dropna().unique().tolist()) if df is not None else ["ALL"]
    sig_filter = st.sidebar.selectbox("Signal", signals)

    sectors = ["ALL"] + sorted(df["Sector"].dropna().unique().tolist()) if df is not None else ["ALL"]
    sec_filter = st.sidebar.selectbox("Sector", sectors)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Triangulation Logic**")
    st.sidebar.markdown("60% DCF + 40% Comps")
    st.sidebar.markdown("BUY > +20% upside")
    st.sidebar.markdown("SELL < −20% upside")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*SOTP exclusions:*")
    st.sidebar.markdown("RELIANCE · ITC · ONGC")
    st.sidebar.markdown("GRASIM · BHARTIARTL · NESTLEIND")

    return sig_filter, sec_filter


# ── UNIVERSE TAB ───────────────────────────────────────────────────────────────
def tab_universe(df, sig_filter, sec_filter):
    # Apply filters
    fdf = df.copy()
    if sig_filter != "ALL":
        fdf = fdf[fdf["Signal"] == sig_filter]
    if sec_filter != "ALL":
        fdf = fdf[fdf["Sector"] == sec_filter]

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Universe", f"{len(df)} stocks")
    with c2: st.metric("BUY",  f"{(df['Signal']=='BUY').sum()}",  delta="↑ upside > 20%")
    with c3: st.metric("HOLD", f"{(df['Signal']=='HOLD').sum()}", delta="±20% band")
    with c4: st.metric("SELL", f"{(df['Signal']=='SELL').sum()}", delta="↓ overvalued")
    with c5: st.metric("N/A",  f"{(df['Signal']=='N/A').sum()}",  delta="SOTP required")

    st.markdown("---")

    # Signal distribution chart
    col_chart, col_sector = st.columns([1, 1])

    with col_chart:
        st.markdown('<div class="section-label">Signal Distribution</div>', unsafe_allow_html=True)
        sig_counts = df["Signal"].value_counts()
        colors = [SIGNAL_COLORS.get(s, "#6b7280") for s in sig_counts.index]
        fig = go.Figure(go.Bar(
            x=sig_counts.index.tolist(), y=sig_counts.values.tolist(),
            marker_color=colors, text=sig_counts.values.tolist(),
            textposition="auto"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=220,
                          xaxis=dict(showgrid=False, color="#4b5563"),
                          yaxis=dict(showgrid=True, gridcolor="#0f172a", color="#4b5563"))
        st.plotly_chart(fig, use_container_width=True)

    with col_sector:
        st.markdown('<div class="section-label">Sector Coverage</div>', unsafe_allow_html=True)
        sec_counts = df["Sector"].value_counts().head(10)
        fig2 = go.Figure(go.Bar(
            x=sec_counts.values.tolist(), y=sec_counts.index.tolist(),
            orientation="h", marker_color="#1d4ed8",
            text=sec_counts.values.tolist(), textposition="auto"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=220,
                           xaxis=dict(showgrid=True, gridcolor="#0f172a", color="#4b5563"),
                           yaxis=dict(showgrid=False, color="#4b5563"))
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: Upside vs WACC, coloured by signal
    st.markdown('<div class="section-label">Upside vs WACC (by signal)</div>', unsafe_allow_html=True)
    plot_df = df.dropna(subset=["Triangulated_Upside", "WACC", "Signal"])
    fig3 = go.Figure()
    for sig, col in SIGNAL_COLORS.items():
        sub = plot_df[plot_df["Signal"] == sig]
        if sub.empty: continue
        fig3.add_trace(go.Scatter(
            x=sub["WACC"] * 100, y=sub["Triangulated_Upside"] * 100,
            mode="markers+text",
            text=sub["Ticker"].str.replace(".NS", "", regex=False),
            textposition="top center", textfont=dict(size=9, color=col),
            marker=dict(color=col, size=8, opacity=0.8),
            name=sig,
            hovertemplate="<b>%{text}</b><br>WACC: %{x:.1f}%<br>Upside: %{y:.1f}%<extra></extra>"
        ))
    fig3.add_hline(y=20,  line_dash="dash", line_color="#22c55e", line_width=0.8, annotation_text="BUY threshold")
    fig3.add_hline(y=-20, line_dash="dash", line_color="#ef4444", line_width=0.8, annotation_text="SELL threshold")
    fig3.update_layout(**PLOTLY_LAYOUT, height=380,
                       xaxis=dict(title="WACC (%)", showgrid=True, gridcolor="#0f172a", color="#4b5563"),
                       yaxis=dict(title="Triangulated Upside (%)", showgrid=True, gridcolor="#0f172a", color="#4b5563"),
                       legend=dict(bgcolor="#0d1321", bordercolor="#1f2937", borderwidth=0.5))
    st.plotly_chart(fig3, use_container_width=True)

    # Main table
    st.markdown('<div class="section-label">Full Universe Table</div>', unsafe_allow_html=True)
    display_cols = {
        "Ticker": "Ticker", "Sector": "Sector", "Price": "CMP (₹)",
        "IntrinsicPrice_DCF": "DCF (₹)", "CompsPrice": "Comps (₹)",
        "Triangulated_Price": "Target (₹)", "Triangulated_Upside": "Upside",
        "Signal": "Signal", "WACC": "WACC", "Beta": "β",
        "PE_Ratio": "P/E", "ROE": "ROE", "BlendedGrowth": "Growth",
        "TerminalMethod": "TV Method", "Comps_Method": "Comps Method"
    }
    existing = {k: v for k, v in display_cols.items() if k in fdf.columns}
    tbl = fdf[list(existing.keys())].rename(columns=existing).copy()

    # Format
    for c in ["CMP (₹)", "DCF (₹)", "Comps (₹)", "Target (₹)"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    for c in ["Upside", "WACC", "ROE", "Growth"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda x: pct(x) if pd.notna(x) else "—")
    for c in ["β", "P/E"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    # Colour rows by signal
    def row_color(row):
        sig = row.get("Signal", "")
        if sig == "BUY":  return ["background-color: rgba(34,197,94,0.05)"] * len(row)
        if sig == "SELL": return ["background-color: rgba(239,68,68,0.05)"] * len(row)
        return [""] * len(row)

    styled = tbl.style.apply(row_color, axis=1)
    st.dataframe(styled, use_container_width=True, height=460)


# ── STOCK DETAIL TAB ───────────────────────────────────────────────────────────
def tab_stock_detail(df):
    tickers = sorted(df["Ticker"].str.replace(".NS", "", regex=False).tolist())
    col_sel, col_info = st.columns([1, 3])

    with col_sel:
        selected = st.selectbox("Select stock", tickers, index=tickers.index("INFY") if "INFY" in tickers else 0)

    row = df[df["Ticker"].str.replace(".NS", "", regex=False) == selected]
    if row.empty:
        row = df[df["Ticker"] == selected + ".NS"]
    if row.empty:
        st.warning("Stock not found.")
        return
    s = row.iloc[0]

    with col_info:
        sig = s.get("Signal", "N/A")
        name = s.get("Ticker", selected)
        sector = s.get("Sector", "")
        st.markdown(f"### {name} &nbsp;&nbsp; {signal_badge(sig)}", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#4b5563;font-size:11px'>{sector} · {s.get('TerminalMethod','—')} terminal · {s.get('Comps_Method','—')} comps</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Top metrics row ─────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    price = s.get("Price", np.nan)
    dcf   = s.get("IntrinsicPrice_DCF", np.nan)
    comps = s.get("CompsPrice", np.nan)
    tri   = s.get("Triangulated_Price", np.nan)
    up    = s.get("Triangulated_Upside", np.nan)
    wacc  = s.get("WACC", np.nan)

    with m1: st.metric("CMP", inr(price))
    with m2: st.metric("DCF Value", inr(dcf), delta=pct((dcf-price)/price) if pd.notna(dcf) and price else None)
    with m3: st.metric("Comps Value", inr(comps), delta=pct((comps-price)/price) if pd.notna(comps) and price else None)
    with m4: st.metric("Triangulated", inr(tri), delta=pct(up) if pd.notna(up) else None)
    with m5: st.metric("WACC", pct(wacc) if pd.notna(wacc) else "—")
    with m6: st.metric("Beta", f"{s.get('Beta', np.nan):.2f}" if pd.notna(s.get('Beta')) else "—")

    st.markdown("---")

    # ── Valuation bar chart ──────────────────────────────────────────────────
    col_bar, col_wacc = st.columns([1, 1])

    with col_bar:
        st.markdown('<div class="section-label">Valuation Summary</div>', unsafe_allow_html=True)
        vals = {"CMP": price, "DCF": dcf, "Comps": comps, "Triangulated": tri}
        valid = {k: v for k, v in vals.items() if pd.notna(v)}
        if valid:
            bar_colors = {"CMP":"#4b5563","DCF":"#3b82f6","Comps":"#8b5cf6","Triangulated":"#10b981"}
            fig = go.Figure(go.Bar(
                x=list(valid.keys()), y=list(valid.values()),
                marker_color=[bar_colors[k] for k in valid],
                text=[f"₹{v:,.0f}" for v in valid.values()],
                textposition="auto"
            ))
            fig.add_hline(y=price, line_dash="dot", line_color="#9ca3af", line_width=1,
                          annotation_text="CMP", annotation_font_color="#9ca3af")
            fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                              yaxis=dict(showgrid=True, gridcolor="#0f172a", color="#4b5563"),
                              xaxis=dict(showgrid=False, color="#4b5563"))
            st.plotly_chart(fig, use_container_width=True)

    with col_wacc:
        st.markdown('<div class="section-label">WACC Build-up</div>', unsafe_allow_html=True)
        ke = s.get("Ke", np.nan); kd = s.get("Kd_AfterTax", np.nan)
        we = s.get("We", np.nan); wd = s.get("Wd", np.nan)
        rf = 0.0480; erp = 0.0746
        wacc_components = {
            "Rf": rf, "Beta×ERP": s.get("Beta", 1.0) * erp,
        }
        if pd.notna(ke):
            fig2 = go.Figure(go.Waterfall(
                x=["Rf (4.80%)", f"β×ERP ({s.get('Beta',1):.2f}×7.46%)",
                   f"Ke={pct(ke)}", f"Kd={pct(kd)}", f"WACC={pct(wacc)}"],
                y=[rf, s.get("Beta",1.0)*erp, -(s.get("Beta",1.0)*erp)*(wd if pd.notna(wd) else 0),
                   (kd if pd.notna(kd) else 0)*(wd if pd.notna(wd) else 0), 0],
                measure=["relative","relative","relative","relative","total"],
                connector=dict(line=dict(color="#1f2937")),
                increasing=dict(marker_color="#3b82f6"),
                decreasing=dict(marker_color="#6b7280"),
                totals=dict(marker_color="#10b981"),
                text=[pct(rf), pct(s.get("Beta",1.0)*erp), "—", "—", pct(wacc)],
                textposition="auto"
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                               yaxis=dict(showgrid=True, gridcolor="#0f172a", color="#4b5563",
                                          tickformat=".1%"),
                               xaxis=dict(showgrid=False, color="#4b5563"))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Three column detail ──────────────────────────────────────────────────
    cd1, cd2, cd3 = st.columns(3)

    with cd1:
        st.markdown('<div class="section-label">Cost of Capital</div>', unsafe_allow_html=True)
        rows = [
            ("Risk-free rate (Rf)", "4.80%"),
            ("ERP (Damodaran Jul 2025)", "7.46%"),
            (f"Beta (5Y weekly OLS)", f"{s.get('Beta', np.nan):.4f}" if pd.notna(s.get('Beta')) else "—"),
            ("Cost of Equity (Ke)", pct(s.get("Ke"), 2)),
            ("ICR", f"{s.get('ICR', np.nan):.1f}x" if pd.notna(s.get('ICR')) and s.get('ICR',0)>0 else "N/A"),
            ("Synthetic spread", pct(s.get("SyntheticSpread"), 2)),
            ("Kd (after-tax)", pct(s.get("Kd_AfterTax"), 2)),
            ("Weight equity (We)", pct(s.get("We"), 1)),
            ("Weight debt (Wd)", pct(s.get("Wd"), 1)),
            ("**WACC**", f"**{pct(s.get('WACC'), 2)}**"),
        ]
        for label, val in rows:
            c_l, c_r = st.columns([2, 1])
            c_l.markdown(f"<span style='font-size:11px;color:#4b5563'>{label}</span>", unsafe_allow_html=True)
            c_r.markdown(f"<span style='font-size:11px;color:#9ca3af;font-family:monospace'>{val}</span>", unsafe_allow_html=True)

    with cd2:
        st.markdown('<div class="section-label">Growth & DCF Bridge</div>', unsafe_allow_html=True)
        pv_fcff = s.get("PV_ExplicitFCFF_Cr", np.nan)
        pv_tv   = s.get("PV_Terminal_Cr", np.nan)
        ev      = s.get("EV_Cr", np.nan)
        tv_pct  = f"{pv_tv/(pv_fcff+pv_tv)*100:.0f}%" if pd.notna(pv_fcff) and pd.notna(pv_tv) and (pv_fcff+pv_tv)>0 else "—"

        rows2 = [
            ("Fundamental growth", pct(s.get("FundamentalGrowth"), 1)),
            ("Market-implied growth", pct(s.get("ImpliedGrowth"), 1)),
            ("**Blended growth (50/50)**", f"**{pct(s.get('BlendedGrowth'), 1)}**"),
            ("Terminal growth", "5.75%" if s.get("TerminalMethod")=="Gordon Growth" else "5.25%"),
            ("Terminal method", s.get("TerminalMethod", "—")),
            ("—", "—"),
            ("PV explicit FCFFs", inr_cr(pv_fcff) if pd.notna(pv_fcff) else "—"),
            ("PV terminal value", inr_cr(pv_tv) if pd.notna(pv_tv) else "—"),
            (f"TV as % of EV", tv_pct),
            ("Enterprise value", inr_cr(ev) if pd.notna(ev) else "—"),
        ]
        for label, val in rows2:
            if label == "—": st.markdown(""); continue
            c_l, c_r = st.columns([2, 1])
            c_l.markdown(f"<span style='font-size:11px;color:#4b5563'>{label}</span>", unsafe_allow_html=True)
            c_r.markdown(f"<span style='font-size:11px;color:#9ca3af;font-family:monospace'>{val}</span>", unsafe_allow_html=True)

    with cd3:
        st.markdown('<div class="section-label">Financials & Multiples</div>', unsafe_allow_html=True)
        rows3 = [
            ("Revenue", inr_cr(s.get("Revenue_Cr"))),
            ("EBITDA", inr_cr(s.get("EBITDA_Cr")) if s.get("EBITDA_Cr",0)>0 else "N/A"),
            ("EBIT", inr_cr(s.get("EBIT_Cr")) if s.get("EBIT_Cr",0)>0 else "N/A"),
            ("Market cap", inr_cr(s.get("MarketCap_Cr"))),
            ("—", "—"),
            ("EPS (TTM)", f"₹{s.get('EPS_TTM', np.nan):.2f}" if pd.notna(s.get("EPS_TTM")) else "—"),
            ("DPS", f"₹{s.get('DPS', 0):.2f}"),
            ("Book value/share", f"₹{s.get('BookValue', np.nan):.0f}" if pd.notna(s.get("BookValue")) else "—"),
            ("Trailing P/E", f"{s.get('PE_Ratio', np.nan):.1f}x" if pd.notna(s.get("PE_Ratio")) else "—"),
            ("ROE", pct(s.get("ROE"), 1)),
            ("ROA", pct(s.get("ROA"), 1)),
            ("Profit margin", pct(s.get("ProfitMargin"), 1)),
            ("Net debt/EBITDA", f"{s.get('NetDebt_EBITDA', np.nan):.2f}x" if pd.notna(s.get("NetDebt_EBITDA")) else "N/A"),
            ("Comps method", s.get("Comps_Method", "—")),
            ("No. of peers", f"{int(s.get('N_Peers', 0))} peers"),
        ]
        for label, val in rows3:
            if label == "—": st.markdown(""); continue
            c_l, c_r = st.columns([2, 1])
            c_l.markdown(f"<span style='font-size:11px;color:#4b5563'>{label}</span>", unsafe_allow_html=True)
            c_r.markdown(f"<span style='font-size:11px;color:#9ca3af;font-family:monospace'>{val}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Sensitivity Grid ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Sensitivity Grid — WACC × Terminal Growth → Intrinsic Price</div>', unsafe_allow_html=True)

    if pd.notna(s.get("WACC")) and pd.notna(s.get("NOPAT_Cr")) and s.get("NOPAT_Cr", 0) > 0:
        shares = s.get("MarketCap_Cr", 1) * 1e7 / s.get("Price", 1) if s.get("Price", 0) > 0 else 1
        sens = sensitivity_grid(
            wacc=s["WACC"],
            growth=s.get("BlendedGrowth", 0.08),
            nopat_cr=s["NOPAT_Cr"],
            total_debt_cr=s.get("TotalDebt_Cr", 0),
            total_cash_cr=s.get("TotalCash_Cr", 0),
            shares=shares
        )

        # Style: green if price > CMP+20%, red if < CMP-20%
        def color_cell(val):
            if pd.isna(val): return "color: #1f2937"
            if val > price * 1.20: return "color: #22c55e; font-weight: 700"
            if val < price * 0.80: return "color: #ef4444"
            return "color: #f59e0b"

        styled_sens = sens.style.applymap(color_cell).format(
            lambda x: f"₹{int(x):,}" if not pd.isna(x) else "—"
        )
        st.dataframe(styled_sens, use_container_width=True)

        st.caption("Green = BUY territory (>+20% vs CMP) · Yellow = HOLD · Red = SELL · **Bold** = base case cell")
    else:
        st.info("Sensitivity grid unavailable — NOPAT not computed (bank/insurance/SOTP stock).")

    # ── Rationale ────────────────────────────────────────────────────────────
    rationale = s.get("Rationale", "")
    if rationale:
        st.markdown("---")
        st.markdown('<div class="section-label">Valuation Rationale</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='background:#0d1321;border:0.5px solid #1f2937;border-radius:6px;padding:12px 16px;font-size:12px;color:#6b7280;line-height:1.8'>{rationale}</div>", unsafe_allow_html=True)


# ── SOTP TAB ───────────────────────────────────────────────────────────────────
def tab_sotp():
    st.markdown("### SOTP Stubs — Structurally Complex Stocks")
    st.info("These 6 stocks are excluded from the standard DCF because a single-model approach is methodologically inappropriate. The stubs below document the correct approach for each.")
    st.markdown("---")

    sotp_stocks = {
        "RELIANCE.NS": {
            "name": "Reliance Industries Ltd",
            "why": "Conglomerate with 5 distinct business verticals. Each segment has different growth, margin, and capital structure profiles. Single-model DCF would average across incompatible assumptions.",
            "method": "SOTP (Sum of the Parts)",
            "segments": [
                ("O2C (Oil-to-Chemicals)", "EV/EBITDA · peer: IOC, BPCL", "~6-7x EBITDA"),
                ("Retail (Reliance Retail)", "P/Sales or DCF · peer: DMart, Avenue Supermarts", "~3-4% margins"),
                ("Digital (Jio Platforms)", "EV/subscriber · ~$50-60/subscriber or DCF", "~400M subscribers"),
                ("E&P (Upstream Oil)", "NAV-based · reserve life × netback", "KG-D6 basin"),
                ("New Energy (Green H2, Solar)", "DCF on capex deployment timeline", "Early stage"),
            ],
            "note": "Aggregate: Sum segment EVs → deduct consolidated net debt → divide by shares."
        },
        "ITC.NS": {
            "name": "ITC Ltd",
            "why": "Cigarettes (high-moat, 80%+ EBIT), Hotels, FMCG (loss-making), Agribusiness, Paper — each demands a different multiple.",
            "method": "SOTP",
            "segments": [
                ("Cigarettes", "P/E · global tobacco peers: BAT, PMI", "~18-22x PE historically"),
                ("Hotels", "EV/EBITDA · peer: IHCL, EIH", "~15-18x EBITDA"),
                ("FMCG (non-cigarettes)", "P/Sales (loss-making) · peer: HUL, Dabur", "~2-3x revenue"),
                ("Agribusiness", "EV/EBITDA · commodity segment", "~8-10x EBITDA"),
                ("Paper & Packaging", "EV/EBITDA · peer: TNPL, Ballarpur", "~6-8x EBITDA"),
            ],
            "note": "ITC has ~₹20,000 Cr net cash — add back at face value to equity value."
        },
        "ONGC.NS": {
            "name": "Oil & Natural Gas Corp",
            "why": "E&P company — value is driven by reserves, not earnings. Standard DCF on accounting earnings ignores reserve depletion and commodity price optionality.",
            "method": "NAV (Net Asset Value) — Reserve-Based",
            "segments": [
                ("Proved developed reserves (PD)", "DCF on production × (price – opex) × reserve life", "2P reserves ~5.9 Bn BOE"),
                ("Proved undeveloped (PUD)", "Risked NAV: 70-80% of PD value", "Exploration upside"),
                ("Subsidiaries (HPCL, OVL)", "Market value of stakes", "Listed: mark-to-market"),
                ("Net debt adjustment", "Deduct: gross debt less surplus cash", "₹~80,000 Cr net debt"),
            ],
            "note": "NAV methodology: Net Present Value of all reserves at long-run oil price ($70-75/bbl) discounted at WACC. Damodaran's oil company NAV spreadsheet is the reference."
        },
        "GRASIM.NS": {
            "name": "Grasim Industries Ltd",
            "why": "Holds ~52% of UltraTech Cement (India's largest cement co) + standalone VSF/Chemicals + Birla Paints (new, loss-making). Cross-holding makes consolidated P&L misleading.",
            "method": "SOTP + Holdco Discount",
            "segments": [
                ("UltraTech stake (52%)", "Market value of listed stake (mark-to-market)", "Apply 20-25% holdco discount"),
                ("VSF & Chemicals (standalone)", "EV/EBITDA · peer: Lenzing, Aditya Birla Nuvo", "~8-10x EBITDA"),
                ("Birla Paints", "P/Sales (investment phase) · peer: Asian Paints, Berger", "~3-5% revenue share"),
            ],
            "note": "Holdco discount of 20-25% is standard for Indian holding companies (Damodaran India 2024 study)."
        },
        "BHARTIARTL.NS": {
            "name": "Bharti Airtel Ltd",
            "why": "Telecom with high operating leverage — EV/EBITDA is the right multiple (not P/E, which is distorted by D&A on massive capex). Also has Africa (AMN) and payments bank operations.",
            "method": "EV/EBITDA + EV/Subscriber",
            "segments": [
                ("India Mobile", "EV/EBITDA · peer: Reliance Jio (unlisted), Vodafone Idea", "Target: ~8-10x EBITDA"),
                ("India Homes (Broadband/DTH)", "EV/subscriber · ~₹5,000-7,000/sub", "~7M broadband subs"),
                ("Africa (Airtel Africa AMN)", "Market cap of listed entity × Airtel's stake", "Listed on NSE + LSE"),
                ("Airtel Payments Bank", "P/GMV or DCF — early stage", "Not separately listed"),
            ],
            "note": "Airtel has ~₹2.1L Cr net debt post-AGR. High leverage makes P/E meaningless — EBITDA is the correct anchor."
        },
        "NESTLEIND.NS": {
            "name": "Nestlé India Ltd",
            "why": "Pure FMCG — appropriate valuation is P/E reversion and EV/EBITDA against global FMCG comps (Nestlé SA, Unilever). DCF terminal growth assumptions are highly sensitive for compounders.",
            "method": "P/E Reversion + EV/EBITDA Comps",
            "segments": [
                ("P/E Reversion", "Median peer P/E · global FMCG · HUL, Dabur, Marico, Nestlé SA", "~55-65x trailing PE"),
                ("EV/EBITDA", "Peer: ~35-40x EBITDA (premium FMCG)", "High moat, pricing power"),
            ],
            "note": "For Nestlé, DCF can be used as a cross-check only. The market prices it as a perpetuity compounder — P/E and EV/EBITDA are the primary anchors."
        },
    }

    for ticker, info in sotp_stocks.items():
        with st.expander(f"**{ticker.replace('.NS','')}** — {info['name']} | Method: {info['method']}"):
            st.markdown(f"**Why excluded from standard DCF:** {info['why']}")
            st.markdown(f"**Recommended Method:** `{info['method']}`")
            st.markdown("**Segment breakdown:**")
            seg_df = pd.DataFrame(info["segments"], columns=["Segment", "Methodology", "Key Metric"])
            st.dataframe(seg_df, use_container_width=True, hide_index=True)
            st.info(f"📌 {info['note']}")


# ── BACKTESTING TAB ────────────────────────────────────────────────────────────
def tab_backtest(df):
    st.markdown("### Backtesting Module — Signal vs. Actual Returns")
    st.markdown("""
    This module validates ILGC signals retrospectively.
    It fetches **actual 12-month returns** for each Nifty 50 stock,
    then compares them against the pipeline signal (BUY/HOLD/SELL)
    to measure whether the model added alpha.
    """)

    if st.button("▶ Run Backtest (fetches live price data via yfinance)", type="primary"):
        import yfinance as yf
        with st.spinner("Fetching 12-month price data for all stocks..."):
            results = []
            progress = st.progress(0)
            tickers = df["Ticker"].tolist()

            for i, ticker in enumerate(tickers):
                try:
                    hist = yf.download(ticker, period="1y", interval="1mo",
                                       progress=False, auto_adjust=True)
                    if len(hist) >= 2:
                        price_start = hist["Close"].iloc[0]
                        price_end   = hist["Close"].iloc[-1]
                        # Handle multi-index (yfinance quirk)
                        if hasattr(price_start, "values"):
                            price_start = float(price_start.values[0])
                            price_end   = float(price_end.values[0])
                        actual_return = (float(price_end) - float(price_start)) / float(price_start)
                        signal = df[df["Ticker"] == ticker]["Signal"].values
                        signal = signal[0] if len(signal) > 0 else "N/A"
                        results.append({
                            "Ticker": ticker.replace(".NS",""),
                            "Signal": signal,
                            "Actual_12M_Return": actual_return,
                            "Start_Price": round(float(price_start), 2),
                            "End_Price": round(float(price_end), 2),
                        })
                except Exception:
                    pass
                progress.progress((i+1)/len(tickers))

        if not results:
            st.error("No data fetched. Check internet / yfinance access.")
            return

        bt_df = pd.DataFrame(results)
        st.success(f"Backtest complete: {len(bt_df)} stocks.")

        # ── Summary statistics ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">Signal Performance Summary</div>', unsafe_allow_html=True)

        for sig in ["BUY", "HOLD", "SELL"]:
            sub = bt_df[bt_df["Signal"] == sig]["Actual_12M_Return"]
            if len(sub) > 0:
                avg = sub.mean() * 100
                med = sub.median() * 100
                hit = (sub > 0).sum() if sig == "BUY" else (sub < 0).sum() if sig == "SELL" else None
                col = SIGNAL_COLORS[sig]
                st.markdown(
                    f"<div style='padding:8px 14px;background:rgba(0,0,0,0.3);border-left:3px solid {col};border-radius:4px;margin-bottom:8px'>"
                    f"<span style='color:{col};font-weight:700;font-family:monospace'>{sig}</span> "
                    f"<span style='color:#6b7280;font-size:12px;margin-left:12px'>"
                    f"Avg return: <b style='color:#d1d5db'>{avg:+.1f}%</b> · "
                    f"Median: <b style='color:#d1d5db'>{med:+.1f}%</b> · "
                    f"n={len(sub)}"
                    f"{'  · Hit rate: <b style=color:#d1d5db>' + str(hit) + '/' + str(len(sub)) + '</b>' if hit is not None else ''}"
                    f"</span></div>",
                    unsafe_allow_html=True
                )

        # ── Scatter: signal vs actual return ────────────────────────────────
        st.markdown('<div class="section-label">Signal vs. Actual 12-Month Return</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for sig, col in SIGNAL_COLORS.items():
            sub = bt_df[bt_df["Signal"] == sig]
            if sub.empty: continue
            fig.add_trace(go.Box(
                y=sub["Actual_12M_Return"] * 100,
                name=sig, marker_color=col,
                boxpoints="all", jitter=0.3, pointpos=-1.5,
                line=dict(color=col, width=1.5),
                fillcolor=f"rgba({','.join(str(int(col[i:i+2],16)) for i in (1,3,5))},0.1)"
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          yaxis=dict(title="Actual 12M Return (%)", showgrid=True,
                                     gridcolor="#0f172a", color="#4b5563", tickformat=".0f"),
                          xaxis=dict(showgrid=False, color="#4b5563"))
        st.plotly_chart(fig, use_container_width=True)

        # ── Full backtest table ──────────────────────────────────────────────
        st.markdown('<div class="section-label">Full Backtest Table</div>', unsafe_allow_html=True)
        bt_df["Actual_12M_Return_%"] = (bt_df["Actual_12M_Return"] * 100).round(1)
        bt_df["Correct?"] = bt_df.apply(
            lambda r: "✓" if (r["Signal"]=="BUY" and r["Actual_12M_Return"]>0)
                         or (r["Signal"]=="SELL" and r["Actual_12M_Return"]<0)
                         else ("~" if r["Signal"]=="HOLD" else "✗"),
            axis=1
        )

        display = bt_df[["Ticker","Signal","Start_Price","End_Price","Actual_12M_Return_%","Correct?"]].copy()
        display.columns = ["Ticker","Signal","Price (Start)","Price (End)","12M Return (%)","Signal Correct?"]

        def color_return(val):
            try:
                v = float(val)
                if v > 10: return "color: #22c55e"
                if v < -10: return "color: #ef4444"
                return "color: #f59e0b"
            except: return ""

        styled_bt = display.style.applymap(color_return, subset=["12M Return (%)"])
        st.dataframe(styled_bt, use_container_width=True, height=400)

        # Save backtest results
        bt_df.to_csv("ilgc_backtest_results.csv", index=False)
        st.success("Saved to ilgc_backtest_results.csv")

    else:
        st.markdown("""
        <div style='background:#0d1321;border:0.5px solid #1f2937;border-radius:8px;padding:16px 20px;font-size:12px;color:#4b5563;line-height:2'>
        <b style='color:#6b7280'>What this module does:</b><br>
        · Fetches 12-month monthly closing prices for every stock via yfinance<br>
        · Computes actual total return: (end price − start price) / start price<br>
        · Compares against the ILGC signal: BUY stocks should have +ve returns, SELL stocks −ve<br>
        · Reports hit rate: BUY signal correct if actual return > 0%<br>
        · Reports hit rate: SELL signal correct if actual return &lt; 0%<br>
        · Box plot shows return distribution by signal bucket<br><br>
        <b style='color:#6b7280'>Limitation:</b> Signals are generated on current prices, not on the prices at signal generation time.
        For a true backtest, you'd need to store timestamped signals and re-run the pipeline historically.
        This module is a <i>contemporaneous validation</i>, not a strict historical backtest.
        </div>
        """, unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("# ILGC &nbsp;·&nbsp; Equity Valuation Dashboard", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#1f2937;margin-top:-12px;margin-bottom:8px'>Integrated Large-cap Growth & Comps · Nifty 50 · DCF + Comps + Triangulation · v9</div>", unsafe_allow_html=True)

    df = load_data()
    sig_filter, sec_filter = sidebar(df)

    if df is None:
        st.warning("No pipeline data found. Run `ilgc_dcf_nifty50_final.py` first to generate `ilgc_full_dashboard_data.json`.")
        st.markdown("```bash\npython ilgc_dcf_nifty50_final.py\n# or on Kaggle: exec(open('ilgc_dcf_nifty50_final.py').read())\n```")
        st.stop()

    tabs = st.tabs(["🌐 Universe", "🔍 Stock Detail", "🏗 SOTP Stubs", "📈 Backtest"])

    with tabs[0]: tab_universe(df, sig_filter, sec_filter)
    with tabs[1]: tab_stock_detail(df)
    with tabs[2]: tab_sotp()
    with tabs[3]: tab_backtest(df)


if __name__ == "__main__":
    main()
