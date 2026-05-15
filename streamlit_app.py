# =============================================================================
# ILGC — Streamlit Community Cloud App
# =============================================================================

# ── Dependency bootstrap (runs before any other import) ──────────────────────
# Streamlit Cloud sometimes misses requirements.txt — this guarantees install.
import subprocess, sys

def _install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

try:
    import plotly
except ImportError:
    _install("plotly>=5.18.0")

try:
    import yfinance
except ImportError:
    _install("yfinance>=0.2.36")

try:
    import statsmodels
except ImportError:
    _install("statsmodels>=0.14.0")

try:
    import tqdm
except ImportError:
    _install("tqdm>=4.66.0")
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, time

# ── Import your pipeline functions directly ───────────────────────────────────
# This works because both files are in the same repo directory.
# Only specific functions are imported — main() is NOT called.
import yfinance as yf
from ilgc_dcf_nifty50_final import (
    get_nifty50_tickers,
    compute_dcf_single,
    run_comps_engine,
    triangulate,
    sensitivity_grid,
    CONFIG,
    STRUCTURAL_EXCLUSIONS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ILGC · Equity Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #060c18; color: #d1d5db; }
[data-testid="stSidebar"] { background-color: #080f1e; }
h1 { color: #3b82f6 !important; font-family: monospace !important; font-size: 20px !important; letter-spacing: 0.1em; }
h2, h3 { color: #6b7280 !important; font-family: monospace !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] { background:#0d1321; border:0.5px solid #1f2937; border-radius:8px; padding:12px; }
[data-testid="stMetricValue"] > div { color:#e5e7eb !important; font-family:monospace; font-size:18px; font-weight:700; }
[data-testid="stMetricLabel"] > div { color:#6b7280 !important; font-size:11px; }
.section-label { font-size:9px; font-weight:700; color:#374151; letter-spacing:0.12em;
  text-transform:uppercase; border-bottom:0.5px solid #1f2937; padding-bottom:4px; margin:14px 0 8px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_BG = dict(paper_bgcolor="#060c18", plot_bgcolor="#060c18",
                 font=dict(family="monospace", color="#6b7280", size=11),
                 margin=dict(l=40, r=20, t=30, b=40))
SIG_COL = {"BUY":"#22c55e", "HOLD":"#f59e0b", "SELL":"#ef4444", "N/A":"#6b7280"}
SIG_BG  = {"BUY":"rgba(34,197,94,0.12)", "HOLD":"rgba(245,158,11,0.12)",
           "SELL":"rgba(239,68,68,0.12)", "N/A":"rgba(107,114,128,0.08)"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{'+' if v>=0 else ''}{v*100:.{d}f}%"

def inr_cr(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if abs(v) >= 1e5: return f"₹{v/1e5:.1f}L Cr"
    return f"₹{v:,.0f} Cr"

def badge(sig):
    col = SIG_COL.get(sig, "#6b7280"); bg = SIG_BG.get(sig, "rgba(0,0,0,0)")
    return f'<span style="background:{bg};color:{col};padding:2px 10px;border-radius:4px;font-weight:700;font-size:11px">{sig}</span>'

# ── Pipeline runner (cached 6 hours so it doesn't re-run on every page load) ─
@st.cache_data(ttl=6*3600, show_spinner=False)
def run_full_pipeline():
    """
    Runs the complete ILGC pipeline and returns three DataFrames.
    Called once per session (or once every 6 hours on Streamlit Cloud).
    """
    tickers, sector_map = get_nifty50_tickers()
    tickers_clean = [t for t in tickers if t not in STRUCTURAL_EXCLUSIONS]

    # Market data for beta calculation
    mkt = yf.download(CONFIG["MARKET_TICKER"], period="5y",
                      interval="1wk", progress=False, auto_adjust=True)
    market_prices = mkt["Close"].squeeze()

    # DCF — run sequentially (Streamlit Cloud has limited CPU; threads cause issues)
    dcf_results = []
    progress_bar = st.progress(0, text="Running DCF pipeline...")
    for i, ticker in enumerate(tickers_clean):
        result = compute_dcf_single(ticker, market_prices, sector_map)
        if result:
            dcf_results.append(result)
        progress_bar.progress((i + 1) / len(tickers_clean),
                              text=f"DCF: {ticker} ({i+1}/{len(tickers_clean)})")
    progress_bar.empty()

    if not dcf_results:
        st.error("Pipeline produced no results. yfinance may be unavailable.")
        st.stop()

    dcf_df   = pd.DataFrame(dcf_results)
    comps_df = run_comps_engine(dcf_df)
    tri_df   = triangulate(dcf_df, comps_df)

    # Merge into one wide DataFrame for the dashboard
    full = dcf_df.merge(comps_df, on="Ticker", how="left") \
                 .merge(tri_df[["Ticker", "Triangulated_Price",
                                "Triangulated_Upside", "Signal", "Blend_Note"]],
                        on="Ticker", how="left")
    return full

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ILGC v9")
    st.markdown("Nifty 50 · DCF + Comps + Triangulation")
    st.markdown("---")
    st.markdown("`Rf`  4.80% — G-sec minus default spread")
    st.markdown("`ERP` 7.46% — Damodaran Jul 2025")
    st.markdown("`Rm`  12.26%")
    st.markdown("---")

    if st.button("🔄 Refresh pipeline data", help="Re-runs the full DCF + Comps pipeline. Takes ~5 min."):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    sig_filter = st.selectbox("Filter by signal", ["ALL", "BUY", "HOLD", "SELL", "N/A"])
    sec_options = ["ALL"]   # filled after data loads
    sec_placeholder = st.empty()

    st.markdown("---")
    st.markdown("**Triangulation:** 60% DCF + 40% Comps")
    st.markdown("**BUY** > +20% · **SELL** < −20%")
    st.markdown("---")
    st.markdown("*SOTP exclusions:*")
    st.caption("RELIANCE · ITC · ONGC · GRASIM · BHARTIARTL · NESTLEIND")

# ── Load data ─────────────────────────────────────────────────────────────────
st.markdown("# ILGC · Equity Valuation Dashboard")
st.markdown("<div style='font-size:11px;color:#374151;margin-top:-10px;margin-bottom:12px'>"
            "Integrated Large-cap Growth & Comps · Nifty 50 · v9</div>",
            unsafe_allow_html=True)

with st.spinner("Loading pipeline data (first load takes ~5 min, then cached for 6 hours)..."):
    df = run_full_pipeline()

# Fill sector filter now that we have data
sectors = ["ALL"] + sorted(df["Sector"].dropna().unique().tolist())
sec_filter = sec_placeholder.selectbox("Filter by sector", sectors)

# Apply filters
fdf = df.copy()
if sig_filter != "ALL":
    fdf = fdf[fdf["Signal"] == sig_filter]
if sec_filter != "ALL":
    fdf = fdf[fdf["Sector"] == sec_filter]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_universe, tab_detail, tab_sotp, tab_backtest = st.tabs([
    "🌐 Universe", "🔍 Stock Detail", "🏗 SOTP Stubs", "📈 Backtest"
])

# =============================================================================
# TAB 1 — UNIVERSE
# =============================================================================
with tab_universe:
    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe",  f"{len(df)}")
    c2.metric("BUY",       f"{(df['Signal']=='BUY').sum()}",  delta="+20% upside")
    c3.metric("HOLD",      f"{(df['Signal']=='HOLD').sum()}", delta="±20% band")
    c4.metric("SELL",      f"{(df['Signal']=='SELL').sum()}", delta="overvalued")
    c5.metric("N/A",       f"{(df['Signal']=='N/A').sum()}",  delta="SOTP required")

    st.markdown("---")

    # Signal distribution + sector coverage
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Signal Distribution</div>', unsafe_allow_html=True)
        sig_c = df["Signal"].value_counts()
        fig = go.Figure(go.Bar(
            x=sig_c.index.tolist(), y=sig_c.values.tolist(),
            marker_color=[SIG_COL.get(s, "#6b7280") for s in sig_c.index],
            text=sig_c.values.tolist(), textposition="auto"
        ))
        fig.update_layout(**PLOTLY_BG, height=220,
                          xaxis=dict(showgrid=False, color="#4b5563"),
                          yaxis=dict(showgrid=True, gridcolor="#0f172a", color="#4b5563"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Upside vs WACC</div>', unsafe_allow_html=True)
        pdata = df.dropna(subset=["Triangulated_Upside", "WACC", "Signal"])
        fig2 = go.Figure()
        for sig, col in SIG_COL.items():
            sub = pdata[pdata["Signal"] == sig]
            if sub.empty: continue
            fig2.add_trace(go.Scatter(
                x=sub["WACC"] * 100, y=sub["Triangulated_Upside"] * 100,
                mode="markers+text",
                text=sub["Ticker"].str.replace(".NS", "", regex=False),
                textposition="top center", textfont=dict(size=8, color=col),
                marker=dict(color=col, size=7), name=sig,
                hovertemplate="<b>%{text}</b><br>WACC:%{x:.1f}%  Upside:%{y:.1f}%<extra></extra>"
            ))
        fig2.add_hline(y=20,  line_dash="dash", line_color="#22c55e", line_width=0.8)
        fig2.add_hline(y=-20, line_dash="dash", line_color="#ef4444", line_width=0.8)
        fig2.update_layout(**PLOTLY_BG, height=220,
                           xaxis=dict(title="WACC (%)", color="#4b5563", showgrid=True, gridcolor="#0f172a"),
                           yaxis=dict(title="Upside (%)", color="#4b5563", showgrid=True, gridcolor="#0f172a"),
                           legend=dict(bgcolor="#0d1321", bordercolor="#1f2937", borderwidth=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    # Main table
    st.markdown('<div class="section-label">Full Universe</div>', unsafe_allow_html=True)

    show = fdf[[
        "Ticker", "Sector", "Price", "IntrinsicPrice_DCF", "CompsPrice",
        "Triangulated_Price", "Triangulated_Upside", "Signal",
        "WACC", "Beta", "PE_Ratio", "ROE", "BlendedGrowth",
        "TerminalMethod", "Comps_Method"
    ]].copy().rename(columns={
        "Price":"CMP(₹)", "IntrinsicPrice_DCF":"DCF(₹)", "CompsPrice":"Comps(₹)",
        "Triangulated_Price":"Target(₹)", "Triangulated_Upside":"Upside",
        "PE_Ratio":"P/E", "BlendedGrowth":"Growth",
        "TerminalMethod":"TV Method", "Comps_Method":"Comps Method"
    })

    for c in ["CMP(₹)", "DCF(₹)", "Comps(₹)", "Target(₹)"]:
        if c in show.columns:
            show[c] = show[c].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    for c in ["Upside", "WACC", "ROE", "Growth"]:
        if c in show.columns:
            show[c] = show[c].apply(lambda x: pct(x) if pd.notna(x) else "—")
    for c in ["Beta", "P/E"]:
        if c in show.columns:
            show[c] = show[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    def row_style(row):
        sig = row.get("Signal","")
        if sig == "BUY":  return ["background-color:rgba(34,197,94,0.05)"]*len(row)
        if sig == "SELL": return ["background-color:rgba(239,68,68,0.05)"]*len(row)
        return [""]*len(row)

    st.dataframe(show.style.apply(row_style, axis=1),
                 use_container_width=True, height=480)

# =============================================================================
# TAB 2 — STOCK DETAIL
# =============================================================================
with tab_detail:
    tickers_list = sorted(df["Ticker"].str.replace(".NS","",regex=False).tolist())
    selected = st.selectbox("Select stock", tickers_list,
                            index=tickers_list.index("INFY") if "INFY" in tickers_list else 0)

    # Match row
    s_row = df[df["Ticker"].str.replace(".NS","",regex=False) == selected]
    if s_row.empty:
        s_row = df[df["Ticker"] == selected + ".NS"]
    if s_row.empty:
        st.warning("Not found."); st.stop()
    s = s_row.iloc[0]

    price = s.get("Price", np.nan)
    dcf   = s.get("IntrinsicPrice_DCF", np.nan)
    comps = s.get("CompsPrice", np.nan)
    tri   = s.get("Triangulated_Price", np.nan)
    up    = s.get("Triangulated_Upside", np.nan)
    sig   = s.get("Signal", "N/A")
    wacc  = s.get("WACC", np.nan)

    # Header
    col_h, col_sig = st.columns([3, 1])
    with col_h:
        st.markdown(f"### {s['Ticker']} — {s.get('Sector','')}")
        st.caption(f"{s.get('TerminalMethod','—')} terminal · {s.get('Comps_Method','—')} comps · {int(s.get('N_Peers',0))} peers")
    with col_sig:
        st.markdown(f"<div style='text-align:right;margin-top:8px'>{badge(sig)}</div>", unsafe_allow_html=True)
        if pd.notna(up):
            col = "#22c55e" if up > 0 else "#ef4444"
            st.markdown(f"<div style='text-align:right;font-family:monospace;font-size:14px;color:{col}'>{pct(up,1)}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Top metrics
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("CMP",         f"₹{price:,.0f}" if pd.notna(price) else "—")
    m2.metric("DCF Value",   f"₹{dcf:,.0f}"   if pd.notna(dcf)   else "—",
              delta=pct((dcf-price)/price)  if pd.notna(dcf) and pd.notna(price) and price else None)
    m3.metric("Comps Value", f"₹{comps:,.0f}" if pd.notna(comps) else "—",
              delta=pct((comps-price)/price) if pd.notna(comps) and pd.notna(price) and price else None)
    m4.metric("Triangulated",f"₹{tri:,.0f}"   if pd.notna(tri)   else "—",
              delta=pct(up) if pd.notna(up) else None)
    m5.metric("WACC",        pct(wacc,2) if pd.notna(wacc) else "—")
    m6.metric("Beta",        f"{s.get('Beta',np.nan):.2f}" if pd.notna(s.get('Beta')) else "—")

    st.markdown("---")

    # Valuation bar + WACC waterfall
    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown('<div class="section-label">Valuation Comparison</div>', unsafe_allow_html=True)
        labels = ["CMP","DCF","Comps","Triangulated"]
        values = [price, dcf, comps, tri]
        colors = ["#4b5563","#3b82f6","#8b5cf6","#10b981"]
        valid  = [(l,v,c) for l,v,c in zip(labels,values,colors) if pd.notna(v)]
        fig3 = go.Figure(go.Bar(
            x=[x[0] for x in valid], y=[x[1] for x in valid],
            marker_color=[x[2] for x in valid],
            text=[f"₹{x[1]:,.0f}" for x in valid], textposition="auto"
        ))
        fig3.add_hline(y=price, line_dash="dot", line_color="#9ca3af", line_width=1)
        fig3.update_layout(**PLOTLY_BG, height=260, showlegend=False,
                           yaxis=dict(showgrid=True, gridcolor="#0f172a", color="#4b5563"),
                           xaxis=dict(showgrid=False, color="#4b5563"))
        st.plotly_chart(fig3, use_container_width=True)

    with cv2:
        st.markdown('<div class="section-label">WACC Build-up</div>', unsafe_allow_html=True)
        ke = s.get("Ke", np.nan); kd = s.get("Kd_AfterTax", np.nan)
        we = s.get("We", np.nan); wd = s.get("Wd", np.nan)
        beta_v = s.get("Beta", 1.0)
        rf_v   = 0.048; erp_v = 0.0746
        if pd.notna(ke):
            fig4 = go.Figure(go.Waterfall(
                x=["Rf","β × ERP","Ke component","Kd component","WACC"],
                y=[rf_v, beta_v*erp_v,
                   -(beta_v*erp_v)*(wd if pd.notna(wd) else 0),
                   (kd if pd.notna(kd) else 0)*(wd if pd.notna(wd) else 0), 0],
                measure=["relative","relative","relative","relative","total"],
                connector=dict(line=dict(color="#1f2937")),
                increasing=dict(marker_color="#3b82f6"),
                decreasing=dict(marker_color="#6b7280"),
                totals=dict(marker_color="#10b981"),
                textposition="auto"
            ))
            fig4.update_layout(**PLOTLY_BG, height=260, showlegend=False,
                               yaxis=dict(tickformat=".1%", showgrid=True,
                                          gridcolor="#0f172a", color="#4b5563"),
                               xaxis=dict(showgrid=False, color="#4b5563"))
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Three-column detail
    d1, d2, d3 = st.columns(3)

    def detail_row(col_obj, label, value):
        a, b = col_obj.columns([2,1])
        a.markdown(f"<span style='font-size:11px;color:#4b5563'>{label}</span>", unsafe_allow_html=True)
        b.markdown(f"<span style='font-size:11px;color:#9ca3af;font-family:monospace'>{value}</span>", unsafe_allow_html=True)

    with d1:
        st.markdown('<div class="section-label">Cost of Capital</div>', unsafe_allow_html=True)
        for label, val in [
            ("Rf", "4.80%"), ("ERP", "7.46% (Damodaran Jul 2025)"),
            ("Beta (5Y OLS)", f"{s.get('Beta',np.nan):.4f}" if pd.notna(s.get('Beta')) else "—"),
            ("Ke", pct(s.get("Ke"), 2)),
            ("ICR", f"{s.get('ICR',0):.1f}x" if s.get('ICR',0)>0 else "N/A (bank)"),
            ("Synthetic spread", pct(s.get("SyntheticSpread"), 2)),
            ("Kd after-tax", pct(s.get("Kd_AfterTax"), 2)),
            ("We / Wd", f"{pct(s.get('We'),0)} / {pct(s.get('Wd'),0)}"),
            ("WACC", pct(s.get("WACC"), 2)),
        ]:
            detail_row(d1, label, val)

    with d2:
        st.markdown('<div class="section-label">Growth & DCF Bridge</div>', unsafe_allow_html=True)
        pv_f = s.get("PV_ExplicitFCFF_Cr", np.nan)
        pv_t = s.get("PV_Terminal_Cr", np.nan)
        tv_pct = f"{pv_t/(pv_f+pv_t)*100:.0f}%" if pd.notna(pv_f) and pd.notna(pv_t) and (pv_f+pv_t)>0 else "—"
        for label, val in [
            ("Fundamental growth",         pct(s.get("FundamentalGrowth"), 1)),
            ("Market-implied growth",      pct(s.get("ImpliedGrowth"), 1)),
            ("Blended growth (50/50)",     pct(s.get("BlendedGrowth"), 1)),
            ("Terminal growth",            "5.75%" if s.get("TerminalMethod")=="Gordon Growth" else "5.25%"),
            ("Terminal method",            s.get("TerminalMethod","—")),
            ("PV explicit FCFFs",          inr_cr(pv_f) if pd.notna(pv_f) else "—"),
            ("PV terminal value",          inr_cr(pv_t) if pd.notna(pv_t) else "—"),
            ("TV as % of EV",              tv_pct),
            ("Enterprise value",           inr_cr(s.get("EV_Cr")) if pd.notna(s.get("EV_Cr")) else "—"),
        ]:
            detail_row(d2, label, val)

    with d3:
        st.markdown('<div class="section-label">Financials & Multiples</div>', unsafe_allow_html=True)
        for label, val in [
            ("Revenue",       inr_cr(s.get("Revenue_Cr"))),
            ("EBITDA",        inr_cr(s.get("EBITDA_Cr")) if s.get("EBITDA_Cr",0)>0 else "N/A"),
            ("Market cap",    inr_cr(s.get("MarketCap_Cr"))),
            ("EPS (TTM)",     f"₹{s.get('EPS_TTM',0):.2f}" if pd.notna(s.get("EPS_TTM")) else "—"),
            ("Book value",    f"₹{s.get('BookValue',0):.0f}" if pd.notna(s.get("BookValue")) else "—"),
            ("Trailing P/E",  f"{s.get('PE_Ratio',np.nan):.1f}x" if pd.notna(s.get("PE_Ratio")) else "—"),
            ("ROE",           pct(s.get("ROE"), 1)),
            ("ROA",           pct(s.get("ROA"), 1)),
            ("Profit margin", pct(s.get("ProfitMargin"), 1)),
            ("Net debt/EBITDA", f"{s.get('NetDebt_EBITDA',np.nan):.2f}x" if pd.notna(s.get("NetDebt_EBITDA")) else "N/A"),
            ("Comps peers",   f"{int(s.get('N_Peers',0))}"),
        ]:
            detail_row(d3, label, val)

    # Sensitivity grid
    st.markdown("---")
    st.markdown('<div class="section-label">Sensitivity Grid — WACC × Terminal Growth → Intrinsic Price</div>', unsafe_allow_html=True)

    nopat = s.get("NOPAT_Cr", np.nan)
    if pd.notna(wacc) and pd.notna(nopat) and nopat > 0:
        shares = s.get("MarketCap_Cr",1)*1e7 / price if price > 0 else 1
        sens = sensitivity_grid(s["Ticker"], s)   # uses the function from the pipeline

        def color_sens(val):
            try:
                v = float(str(val).replace("₹","").replace(",",""))
                if v > price*1.20: return "color:#22c55e;font-weight:700"
                if v < price*0.80: return "color:#ef4444"
                return "color:#f59e0b"
            except: return ""

        st.dataframe(sens, use_container_width=True)
        st.caption("Green >+20% vs CMP (BUY) · Yellow ±20% (HOLD) · Red <-20% (SELL)")
    else:
        st.info("Sensitivity grid not available for this stock (bank, insurance, or SOTP).")

    # Rationale
    rationale = s.get("Rationale","")
    if rationale:
        st.markdown("---")
        st.markdown('<div class="section-label">Valuation Rationale</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#0d1321;border:0.5px solid #1f2937;border-radius:6px;"
            f"padding:12px 16px;font-size:12px;color:#6b7280;line-height:1.8'>{rationale}</div>",
            unsafe_allow_html=True
        )

# =============================================================================
# TAB 3 — SOTP STUBS
# =============================================================================
with tab_sotp:
    st.markdown("### SOTP Stubs — Structurally Complex Stocks")
    st.info("These 6 stocks are excluded from the standard DCF pipeline. Each needs a bespoke approach.")

    sotp = {
        "RELIANCE": ("SOTP — 5 segments", "O2C (EV/EBITDA) + Retail (P/Sales) + Jio (EV/subscriber) + E&P (NAV) + New Energy (DCF)"),
        "ITC":      ("SOTP — 5 segments", "Cigarettes (P/E ~18-22x) + Hotels (EV/EBITDA) + FMCG (P/Sales) + Agri + Paper"),
        "ONGC":     ("NAV — Reserve-based", "PD reserves DCF + PUD risked NAV + listed subsidiaries (HPCL mark-to-market) − net debt"),
        "GRASIM":   ("SOTP + holdco discount", "UltraTech stake (market value × 75%) + VSF/Chemicals (EV/EBITDA) + Birla Paints (P/Sales)"),
        "BHARTIARTL":("EV/EBITDA + EV/subscriber", "India Mobile (8-10x EBITDA) + Broadband (EV/sub) + Africa (Airtel Africa market cap)"),
        "NESTLEIND": ("P/E reversion + EV/EBITDA", "Global FMCG comps: Nestlé SA, HUL, Dabur — ~55-65x trailing P/E, ~35-40x EV/EBITDA"),
    }
    for ticker, (method, detail) in sotp.items():
        with st.expander(f"**{ticker}** — {method}"):
            st.markdown(detail)

# =============================================================================
# TAB 4 — BACKTEST
# =============================================================================
with tab_backtest:
    st.markdown("### Signal Backtest — Actual 12-Month Returns vs ILGC Signals")
    st.markdown("Fetches real 12-month returns for each stock and checks whether BUY/SELL signals were correct.")

    if st.button("▶ Run Backtest", type="primary"):
        results = []
        prog = st.progress(0, text="Fetching price data...")
        tickers = df["Ticker"].tolist()

        for i, ticker in enumerate(tickers):
            try:
                hist = yf.download(ticker, period="1y", interval="1mo",
                                   progress=False, auto_adjust=True)
                if len(hist) >= 2:
                    p0 = float(hist["Close"].iloc[0].values[0]  if hasattr(hist["Close"].iloc[0], "values") else hist["Close"].iloc[0])
                    p1 = float(hist["Close"].iloc[-1].values[0] if hasattr(hist["Close"].iloc[-1],"values") else hist["Close"].iloc[-1])
                    ret = (p1 - p0) / p0
                    sig = df[df["Ticker"]==ticker]["Signal"].values
                    results.append({
                        "Ticker": ticker.replace(".NS",""),
                        "Signal": sig[0] if len(sig)>0 else "N/A",
                        "12M Return": ret,
                        "Start ₹": round(p0,2),
                        "End ₹":   round(p1,2),
                    })
            except Exception:
                pass
            prog.progress((i+1)/len(tickers), text=f"Fetching {ticker}...")
        prog.empty()

        bt = pd.DataFrame(results)
        st.success(f"Backtest complete: {len(bt)} stocks.")

        # Summary per signal
        st.markdown('<div class="section-label">Return by Signal Bucket</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        for col_obj, sig in zip([sc1, sc2, sc3], ["BUY","HOLD","SELL"]):
            sub = bt[bt["Signal"]==sig]["12M Return"]
            if len(sub):
                col_obj.metric(f"{sig} (n={len(sub)})",
                               f"{sub.mean()*100:+.1f}% avg",
                               delta=f"median {sub.median()*100:+.1f}%")

        # Box plot
        fig5 = go.Figure()
        for sig, col in SIG_COL.items():
            sub = bt[bt["Signal"]==sig]
            if sub.empty: continue
            fig5.add_trace(go.Box(
                y=sub["12M Return"]*100, name=sig,
                marker_color=col, line=dict(color=col),
                boxpoints="all", jitter=0.3
            ))
        fig5.update_layout(**PLOTLY_BG, height=360,
                           yaxis=dict(title="12M Return (%)", showgrid=True,
                                      gridcolor="#0f172a", color="#4b5563"),
                           xaxis=dict(color="#4b5563"))
        st.plotly_chart(fig5, use_container_width=True)

        # Table
        bt["12M Return %"] = (bt["12M Return"]*100).round(1)
        bt["Correct?"] = bt.apply(
            lambda r: "✓" if (r["Signal"]=="BUY" and r["12M Return"]>0)
                         or (r["Signal"]=="SELL" and r["12M Return"]<0) else "✗", axis=1)
        st.dataframe(bt[["Ticker","Signal","Start ₹","End ₹","12M Return %","Correct?"]],
                     use_container_width=True, height=400)
        bt.to_csv("/tmp/ilgc_backtest.csv", index=False)
        st.success("Results also saved to ilgc_backtest.csv")
    else:
        st.markdown("""
        **What this does:** Fetches real 12-month closing prices for every Nifty 50 stock via yfinance.
        Computes actual return, compares against your ILGC signal, and reports the hit rate.

        **Limitation:** This is contemporaneous validation — signals are generated on *current* prices,
        not on prices at the time the signal was originally made. A proper historical backtest would
        require timestamped signal snapshots.
        """)

# =============================================================================
# requirements.txt contents (create this file in your repo root):
# =============================================================================
# streamlit>=1.32.0
# plotly>=5.18.0
# pandas>=2.0.0
# numpy>=1.26.0
# yfinance>=0.2.36
# statsmodels>=0.14.0
# requests>=2.31.0
# tqdm>=4.66.0
# =============================================================================
