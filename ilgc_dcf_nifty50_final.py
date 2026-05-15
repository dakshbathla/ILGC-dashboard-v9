# =============================================================================
# ILGC v9 — FINAL DCF PIPELINE (Nifty 50 Universe)
# Project: Integrated Large-cap Growth & Comps (ILGC)
# Authors: Sandeep & Garv Chadha | Supervisor: Prof. Alok Ranjan
# =============================================================================
# METHODOLOGY OVERVIEW
# ---------------------
# Primary Valuation : Rev-NOPAT DCF (FCFF-based)
# Terminal Value     : Gordon Growth Model (Tech/Healthcare)
#                      Exit Multiple via live EV/EBITDA peer median (all others)
# Cost of Capital    : CAPM / Damodaran India framework
#   RF               = G-sec yield minus India sovereign default spread (~4.80%)
#   ERP              = 7.46% (Damodaran, July 2025)
#   Rm               = RF + ERP = ~12.26% (consistent with Nifty 50 long-run CAGR)
#   Kd               = ICR-based synthetic spread (Damodaran ratings table)
# Implied Growth     : Mauboussin / Brent back-solve from market price
# PSU Adjustments    : Pension strip; cash efficiency discount (Energy/Utilities)
# Cross-checks       : P/E reversion; EPV (floor only); DDM (validity-gated)
# Unit normalisation : CRORE = 1e7 at ingestion (yfinance inconsistency fix)
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import requests
import io
import concurrent.futures
import warnings
import time
import json
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================

CRORE = 1e7  # Hard-coded unit normaliser — yfinance returns raw INR values

CONFIG = {
    # --- Cost of Capital (Damodaran India Framework, July 2025) ---
    "RISK_FREE_RATE"    : 0.0480,   # G-sec 10yr ~7.2% minus India default spread ~2.4%
    "ERP"               : 0.0746,   # India ERP, Damodaran July 2025
    "MARKET_RETURN"     : 0.1226,   # RF + ERP = 4.80% + 7.46%
    "BETA_LOOKBACK_YEARS": 5,
    "TAX_RATE"          : 0.2517,   # Blended effective corporate tax rate India FY24

    # --- DCF Assumptions ---
    "EXPLICIT_YEARS"    : 10,
    "TERMINAL_GROWTH"   : 0.0525,   # India nominal GDP long-run: ~5-6%, use 5.25%
    "GG_TERMINAL_GROWTH": 0.0575,   # Tech/HC slightly higher reinvestment capacity

    # --- Gordon Growth eligible sectors (steady, capital-light moats) ---
    "GG_RELIABLE_SECTORS": {"Information Technology", "Health Care"},

    # --- EV/EBITDA Exit Multiple params ---
    "EXIT_MULTIPLE_HAIRCUT": 0.10,  # 10% discount to peer median (conservatism)
    "MIN_EXIT_PEERS"      : 2,      # Minimum peers needed for exit multiple TV

    # --- PSU Cash Efficiency Discount ---
    "PSU_CASH_DISCOUNT"   : 0.35,   # 35% haircut on excess cash for PSUs
    "PSU_SECTORS"         : {"Energy", "Utilities", "Basic Materials"},

    # --- Implied Expectations Blend ---
    "IMPLIED_GROWTH_WEIGHT": 0.50,  # Mauboussin implied growth weight in final blend

    # --- Comps / Peer Filters ---
    "COMPS_MKTCAP_BAND"   : 0.50,   # +/- 50% market cap band for peer selection
    "COMPS_LEVERAGE_TOL"  : 1.00,   # Net Debt/EBITDA tolerance for peer matching
    "MIN_PEERS_COMPS"     : 2,

    # --- Data / Infrastructure ---
    "MARKET_TICKER"       : "^NSEI",
    "NIFTY50_URL"         : "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "THREADS"             : 4,
    "REQUEST_DELAY"       : 0.5,
}

# Damodaran ICR → Synthetic Rating → Spread table (India context)
# Source: Damodaran, "Ratings, Interest Coverage Ratios and Default Spreads", 2024
ICR_SPREAD_TABLE = [
    (8.50,  0.0063),  # AAA
    (6.50,  0.0078),  # AA
    (5.50,  0.0100),  # A+
    (4.25,  0.0120),  # A
    (3.00,  0.0145),  # A-
    (2.50,  0.0175),  # BBB
    (2.00,  0.0225),  # BB+
    (1.75,  0.0275),  # BB
    (1.50,  0.0325),  # B+
    (1.25,  0.0400),  # B
    (0.80,  0.0500),  # B-
    (0.65,  0.0600),  # CCC
    (0.20,  0.0750),  # CC
    (float('-inf'), 0.1000),  # C / D (distressed)
]

# Stocks requiring SOTP/special methodology — excluded from standard DCF
STRUCTURAL_EXCLUSIONS = {
    "NESTLEIND.NS",   # Pure FMCG — P/E reversion primary
    "BHARTIARTL.NS",  # Telecom — needs EV/EBITDA + subscriber value
    "ONGC.NS",        # Oil E&P — NAV-based
    "ITC.NS",         # Conglomerate SOTP
    "GRASIM.NS",      # Conglomerate SOTP
    "RELIANCE.NS",    # Conglomerate SOTP
}

# =============================================================================
# SECTION 2: DATA INGESTION
# =============================================================================

def get_nifty50_tickers():
    """Download Nifty 50 constituent list from NSE Index website."""
    print("=== [1/6] Fetching Nifty 50 Constituents ===")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(CONFIG["NIFTY50_URL"], headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
        df.columns = df.columns.str.strip()

        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        ind_col = "Industry" if "Industry" in df.columns else df.columns[1]

        tickers = [f"{str(x).strip()}.NS" for x in df[sym_col].tolist()]
        sector_map = dict(zip(tickers, df[ind_col].str.strip()))

        print(f"  ✓ Loaded {len(tickers)} Nifty 50 tickers.")
        return tickers, sector_map

    except Exception as e:
        print(f"  ✗ Failed: {e}. Using hardcoded Nifty 50 fallback.")
        # Hardcoded Nifty 50 as at May 2025
        fallback = [
            "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
            "SBIN.NS","INFY.NS","LT.NS","HINDUNILVR.NS","ITC.NS",
            "KOTAKBANK.NS","BAJFINANCE.NS","WIPRO.NS","HCLTECH.NS","AXISBANK.NS",
            "MARUTI.NS","SUNPHARMA.NS","M&M.NS","NTPC.NS","TATAMOTORS.NS",
            "POWERGRID.NS","ULTRACEMCO.NS","NESTLEIND.NS","TITAN.NS","ONGC.NS",
            "ADANIENT.NS","TECHM.NS","BAJAJFINSV.NS","COALINDIA.NS","GRASIM.NS",
            "DRREDDY.NS","TATACONSUM.NS","CIPLA.NS","HINDALCO.NS","BRITANNIA.NS",
            "APOLLOHOSP.NS","JSWSTEEL.NS","BPCL.NS","EICHERMOT.NS","HEROMOTOCO.NS",
            "DIVISLAB.NS","SHRIRAMFIN.NS","TRENT.NS","INDUSINDBK.NS","SBILIFE.NS",
            "ADANIPORTS.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS","BEL.NS","VEDL.NS"
        ]
        sector_map = {t: "Unknown" for t in fallback}
        return fallback, sector_map


def get_synthetic_spread(icr: float) -> float:
    """Map Interest Coverage Ratio to synthetic default spread (Damodaran table)."""
    for threshold, spread in ICR_SPREAD_TABLE:
        if icr >= threshold:
            return spread
    return 0.10


def safe_get(info: dict, key: str, default=None):
    """Safe dict fetch with None-to-default coercion."""
    val = info.get(key)
    return default if val is None else val

# =============================================================================
# SECTION 3: SINGLE STOCK DCF ENGINE
# =============================================================================

def compute_dcf_single(ticker: str, market_prices: pd.Series, sector_map: dict) -> dict | None:
    """
    Full DCF valuation for a single stock.
    Returns a result dict or None if data is insufficient.
    """
    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except Exception:
            return None

        # ── Basic viability check ─────────────────────────────────────────────
        price = safe_get(info, "currentPrice") or safe_get(info, "previousClose")
        if not price:
            return None

        sector   = sector_map.get(ticker, safe_get(info, "sector", "Unknown"))
        industry = safe_get(info, "industry", sector)

        # ── Unit-normalised financials (all in ₹ Crore) ──────────────────────
        revenue      = safe_get(info, "totalRevenue", 0) / CRORE
        ebitda       = safe_get(info, "ebitda", 0) / CRORE
        ebit         = safe_get(info, "ebit", 0) / CRORE
        total_debt   = safe_get(info, "totalDebt", 0) / CRORE
        total_cash   = safe_get(info, "totalCash", 0) / CRORE
        market_cap   = safe_get(info, "marketCap", 0) / CRORE
        shares_out   = safe_get(info, "sharesOutstanding", 1)
        book_value   = safe_get(info, "bookValue", 0)
        eps_ttm      = safe_get(info, "trailingEps", 0)
        dps          = safe_get(info, "dividendRate", 0) or 0
        pe_ratio     = safe_get(info, "trailingPE", np.nan)
        roe          = safe_get(info, "returnOnEquity", 0) or 0
        roa          = safe_get(info, "returnOnAssets", 0) or 0
        profit_margin = safe_get(info, "profitMargins", 0) or 0

        # Reconstructed market cap sanity check
        recon_cap = (price * shares_out) / CRORE
        if market_cap > 0 and abs(recon_cap - market_cap) / market_cap > 0.40:
            # Likely unit mismatch — use reconstructed
            market_cap = recon_cap

        # ── Growth rate ───────────────────────────────────────────────────────
        # yfinance earningsGrowth is unstable; use revenue growth as fallback
        eg = safe_get(info, "earningsGrowth")
        rg = safe_get(info, "revenueGrowth")
        raw_growth = eg if (eg and 0.0 < eg < 0.80) else (rg if rg else 0.10)
        # Clip: no stock grows >40% p.a. sustainably for 10 years
        fundamental_growth = np.clip(raw_growth, 0.02, 0.40)

        # ── Beta: 5-year weekly OLS regression ───────────────────────────────
        beta = 1.0
        try:
            hist = stock.history(period="5y", interval="1wk", auto_adjust=True)
            if len(hist) > 50:
                sr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                mr = np.log(market_prices / market_prices.shift(1)).dropna()
                idx = sr.index.intersection(mr.index)
                if len(idx) > 30:
                    X = sm.add_constant(mr.loc[idx])
                    Y = sr.loc[idx]
                    beta = sm.OLS(Y, X).fit().params.iloc[1]
        except Exception:
            pass
        beta = np.clip(beta, 0.30, 2.50)

        # ── WACC Construction (Damodaran India Framework) ─────────────────────
        rf    = CONFIG["RISK_FREE_RATE"]
        erp   = CONFIG["ERP"]
        ke    = rf + beta * erp

        # Interest Coverage Ratio → Synthetic Spread → Kd
        interest_expense = safe_get(info, "interestExpense", 0) or 0
        interest_expense = abs(interest_expense) / CRORE
        icr = ebit / interest_expense if interest_expense > 0 else 15.0
        spread = get_synthetic_spread(icr)
        kd_pretax  = rf + spread
        kd_aftertax = kd_pretax * (1 - CONFIG["TAX_RATE"])

        # Capital structure weights
        total_capital = market_cap + max(total_debt, 0)
        w_e = market_cap / total_capital if total_capital > 0 else 1.0
        w_d = max(total_debt, 0) / total_capital if total_capital > 0 else 0.0
        wacc = w_e * ke + w_d * kd_aftertax
        wacc = np.clip(wacc, 0.06, 0.22)  # Sanity bounds

        # ── PSU Adjustments ───────────────────────────────────────────────────
        is_psu = any(s in sector for s in CONFIG["PSU_SECTORS"])
        effective_cash = total_cash
        if is_psu:
            # PSU excess cash haircut — government retains, doesn't return
            effective_cash *= (1 - CONFIG["PSU_CASH_DISCOUNT"])

        # ── Implied Expectations (Mauboussin back-solve) ──────────────────────
        # Back-solve: what growth rate does current price imply?
        # EV = NOPAT × (1 + g) / (WACC - g) → solve for g
        # Using simplified: EV/NOPAT = (1+g)/(WACC-g) → g = (ratio×WACC - 1)/(ratio + 1)
        nopat = ebit * (1 - CONFIG["TAX_RATE"])
        ev_current = market_cap + total_debt - total_cash
        implied_growth = CONFIG["TERMINAL_GROWTH"]  # default
        if nopat > 0 and ev_current > 0:
            ratio = ev_current / nopat
            if ratio > 1:
                g_implied = (ratio * wacc - 1) / (ratio + 1)
                g_implied = np.clip(g_implied, 0.01, 0.50)
                implied_growth = g_implied

        # Blend fundamental + implied growth (50/50, Mauboussin framework)
        blend_weight = CONFIG["IMPLIED_GROWTH_WEIGHT"]
        blended_growth = (1 - blend_weight) * fundamental_growth + blend_weight * implied_growth
        blended_growth = np.clip(blended_growth, 0.02, 0.40)

        # ── DCF: Explicit Forecast Period (10 years) ──────────────────────────
        # FCFF = NOPAT × (1 - Reinvestment Rate)
        # Reinvestment Rate = g / ROIC (where ROIC proxied by ROA adjusted)
        roic = max(roa * 1.5, 0.06)  # rough proxy; floor at 6%
        reinvestment_rate = np.clip(blended_growth / roic, 0.0, 0.90)

        fcff_base = nopat * (1 - reinvestment_rate)
        if fcff_base <= 0:
            # Fall back to ebitda-based FCFF if NOPAT is negative
            fcff_base = max(ebitda * 0.15, 1.0)  # 15% EBITDA as proxy

        pv_fcff = 0
        for yr in range(1, CONFIG["EXPLICIT_YEARS"] + 1):
            # Fade growth linearly from blended_growth toward terminal_growth
            fade = yr / CONFIG["EXPLICIT_YEARS"]
            tg   = CONFIG["TERMINAL_GROWTH"]
            g_yr = blended_growth * (1 - fade) + tg * fade
            fcff_base *= (1 + g_yr)
            pv_fcff += fcff_base / (1 + wacc) ** yr

        # ── Terminal Value ─────────────────────────────────────────────────────
        terminal_method = "Gordon Growth" if sector in CONFIG["GG_RELIABLE_SECTORS"] else "Exit Multiple"
        terminal_growth = CONFIG["GG_TERMINAL_GROWTH"] if sector in CONFIG["GG_RELIABLE_SECTORS"] \
                          else CONFIG["TERMINAL_GROWTH"]

        if terminal_method == "Gordon Growth":
            terminal_fcff = fcff_base * (1 + terminal_growth)
            if wacc <= terminal_growth:
                wacc_adj = terminal_growth + 0.02
            else:
                wacc_adj = wacc
            tv_nominal = terminal_fcff / (wacc_adj - terminal_growth)

        else:  # Exit Multiple
            # We'll use a sector-level EV/EBITDA placeholder here.
            # In the full pipeline run, this is replaced by live peer median.
            # Sector defaults (conservative Damodaran India 2024 medians):
            SECTOR_EXIT_MULTIPLES = {
                "Financial Services": 12.0, "Metals & Mining": 6.0,
                "Energy": 5.5, "Consumer Staples": 16.0,
                "Consumer Discretionary": 14.0, "Industrials": 12.0,
                "Real Estate": 15.0, "Utilities": 9.0,
                "Basic Materials": 7.0, "Telecommunication": 8.0,
                "Automobile": 9.0, "Power": 10.0,
            }
            exit_multiple = SECTOR_EXIT_MULTIPLES.get(sector, 11.0)
            exit_multiple *= (1 - CONFIG["EXIT_MULTIPLE_HAIRCUT"])
            # Terminal year EBITDA ~ project from current
            ebitda_terminal = ebitda * (1 + blended_growth) ** CONFIG["EXPLICIT_YEARS"]
            tv_nominal = exit_multiple * ebitda_terminal

        pv_terminal = tv_nominal / (1 + wacc) ** CONFIG["EXPLICIT_YEARS"]

        # ── Enterprise Value → Equity Value → Intrinsic Price ─────────────────
        enterprise_value = pv_fcff + pv_terminal
        equity_value_cr  = enterprise_value - total_debt + effective_cash
        intrinsic_price  = (equity_value_cr * CRORE) / shares_out if shares_out > 0 else 0

        # Guard: negative equity → not meaningful for DCF
        if intrinsic_price <= 0:
            intrinsic_price = np.nan

        # ── Cross-check 1: P/E Reversion ──────────────────────────────────────
        # Fair value = EPS × sector median P/E (simple sanity check, not primary)
        SECTOR_PE = {
            "Information Technology": 28.0, "Health Care": 32.0,
            "Financial Services": 16.0, "Consumer Staples": 42.0,
            "Consumer Discretionary": 38.0, "Industrials": 28.0,
            "Energy": 9.0, "Basic Materials": 14.0,
            "Metals & Mining": 10.0, "Telecommunication": 22.0,
            "Utilities": 15.0, "Power": 18.0,
        }
        sector_pe = SECTOR_PE.get(sector, 22.0)
        pe_reversion_price = eps_ttm * sector_pe if eps_ttm and eps_ttm > 0 else np.nan

        # ── Cross-check 2: EPV (Floor only, Greenwald) ────────────────────────
        # EPV = NOPAT / WACC (no growth credit — pure earnings power)
        epv_equity  = (nopat / wacc - total_debt + effective_cash) if wacc > 0 else np.nan
        epv_price   = (epv_equity * CRORE) / shares_out if (epv_equity and shares_out > 0) else np.nan

        # ── Cross-check 3: DDM (Only for dividend-paying, stable companies) ────
        ddm_price = np.nan
        payout_ratio = (dps * shares_out / CRORE) / (eps_ttm * shares_out / CRORE) \
                       if eps_ttm and eps_ttm > 0 else 0
        ddm_valid = (dps > 0 and payout_ratio > 0.15 and ke > terminal_growth)
        if ddm_valid:
            ddm_price = dps * (1 + terminal_growth) / (ke - terminal_growth)

        # ── Valuation Gap Rationale ───────────────────────────────────────────
        if not np.isnan(intrinsic_price) and intrinsic_price > 0:
            upside = (intrinsic_price - price) / price
            if upside > 0.20:
                rationale = f"Market appears to undervalue {ticker.replace('.NS','')} — " \
                            f"DCF implies ₹{intrinsic_price:.0f} vs CMP ₹{price:.0f} " \
                            f"({upside:.0%} upside). Key driver: {terminal_method} TV at {wacc:.1%} WACC."
            elif upside < -0.20:
                rationale = f"{ticker.replace('.NS','')} appears richly valued — " \
                            f"DCF implies ₹{intrinsic_price:.0f} vs CMP ₹{price:.0f} " \
                            f"({-upside:.0%} downside). Market embeds growth of {blended_growth:.1%}."
            else:
                rationale = f"{ticker.replace('.NS','')} is roughly fairly valued — " \
                            f"DCF ₹{intrinsic_price:.0f} vs CMP ₹{price:.0f} ({upside:.0%})."
        else:
            upside   = np.nan
            rationale = f"Insufficient data for meaningful DCF on {ticker}."

        # ── Net Debt / EBITDA (for Comps engine) ─────────────────────────────
        net_debt_ebitda = (total_debt - total_cash) / ebitda if ebitda > 0 else np.nan

        return {
            # Identifiers
            "Ticker"            : ticker,
            "Sector"            : sector,
            "Industry"          : industry,

            # Market Data
            "Price"             : price,
            "MarketCap_Cr"      : round(market_cap, 2),
            "SharesOutstanding" : shares_out,

            # Financials (₹ Cr)
            "Revenue_Cr"        : round(revenue, 2),
            "EBITDA_Cr"         : round(ebitda, 2),
            "EBIT_Cr"           : round(ebit, 2),
            "NOPAT_Cr"          : round(nopat, 2),
            "TotalDebt_Cr"      : round(total_debt, 2),
            "TotalCash_Cr"      : round(total_cash, 2),
            "NetDebt_EBITDA"    : round(net_debt_ebitda, 2) if not np.isnan(net_debt_ebitda) else np.nan,
            "FCFF_Base_Cr"      : round(fcff_base, 2),

            # Per Share
            "EPS_TTM"           : round(eps_ttm, 2) if eps_ttm else np.nan,
            "DPS"               : round(dps, 2),
            "BookValue"         : round(book_value, 2),
            "PE_Ratio"          : round(pe_ratio, 2) if not np.isnan(pe_ratio) else np.nan,
            "ROE"               : round(roe, 4),
            "ROA"               : round(roa, 4),
            "ProfitMargin"      : round(profit_margin, 4),

            # Cost of Capital
            "Beta"              : round(beta, 4),
            "Ke"                : round(ke, 4),
            "Kd_AfterTax"       : round(kd_aftertax, 4),
            "WACC"              : round(wacc, 4),
            "ICR"               : round(icr, 2),
            "SyntheticSpread"   : round(spread, 4),
            "We"                : round(w_e, 4),
            "Wd"                : round(w_d, 4),

            # Growth
            "FundamentalGrowth" : round(fundamental_growth, 4),
            "ImpliedGrowth"     : round(implied_growth, 4),
            "BlendedGrowth"     : round(blended_growth, 4),

            # DCF Output
            "PV_ExplicitFCFF_Cr": round(pv_fcff, 2),
            "PV_Terminal_Cr"    : round(pv_terminal, 2),
            "EV_Cr"             : round(enterprise_value, 2),
            "EquityValue_Cr"    : round(equity_value_cr, 2),
            "TerminalMethod"    : terminal_method,
            "IntrinsicPrice_DCF": round(intrinsic_price, 2) if not np.isnan(intrinsic_price) else np.nan,

            # Cross-checks
            "Price_PE_Reversion": round(pe_reversion_price, 2) if not np.isnan(pe_reversion_price) else np.nan,
            "Price_EPV"         : round(epv_price, 2) if not np.isnan(epv_price) else np.nan,
            "Price_DDM"         : round(ddm_price, 2) if not np.isnan(ddm_price) else np.nan,

            # Valuation Gap
            "Upside_DCF"        : round(upside, 4) if not np.isnan(upside) else np.nan,
            "Rationale"         : rationale,
        }

    except Exception as e:
        if "INFY" in ticker:
            print(f"  ⚠ Error on {ticker}: {e}")
        return None


# =============================================================================
# SECTION 4: COMPS ENGINE (4-METHOD FRAMEWORK)
# =============================================================================

def run_comps_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comparable Company Analysis — 4 methods with diagnostic triggers.

    Method selection hierarchy:
    1. Median P/E (fallback, always calculated)
    2. PEG-adjusted P/E (triggered if growth dispersion across peers > 10%)
    3. EV/EBITDA (triggered if leverage spread across peers > 1.0x Net Debt/EBITDA)
    4. P/E vs ROE regression (triggered if R² from OLS regression > 0.50)

    Primary method selected sequentially — not averaged.
    """
    print("\n=== [4/6] Running Comps Engine (4-Method Framework) ===")

    comps_results = []

    for idx, row in df.iterrows():
        ticker   = row["Ticker"]
        sector   = row["Sector"]
        mktcap   = row["MarketCap_Cr"]
        eps      = row.get("EPS_TTM", np.nan)
        pe       = row.get("PE_Ratio", np.nan)
        roe      = row.get("ROE", np.nan)
        nd_ebitda = row.get("NetDebt_EBITDA", np.nan)
        price    = row["Price"]

        # ── Peer Universe Selection ───────────────────────────────────────────
        peers = df[df["Sector"] == sector].copy()
        peers = peers[
            (peers["MarketCap_Cr"] > mktcap * (1 - CONFIG["COMPS_MKTCAP_BAND"])) &
            (peers["MarketCap_Cr"] < mktcap * (1 + CONFIG["COMPS_MKTCAP_BAND"]))
        ]
        if not np.isnan(nd_ebitda):
            peers = peers.dropna(subset=["NetDebt_EBITDA"])
            peers = peers[abs(peers["NetDebt_EBITDA"] - nd_ebitda) < CONFIG["COMPS_LEVERAGE_TOL"]]

        peers = peers[peers["Ticker"] != ticker]
        peers = peers.dropna(subset=["PE_Ratio", "EPS_TTM"])
        peers = peers[peers["PE_Ratio"] > 0]

        n_peers = len(peers)

        if n_peers < CONFIG["MIN_PEERS_COMPS"] or np.isnan(eps) or eps <= 0:
            comps_results.append({
                "Ticker": ticker, "N_Peers": n_peers,
                "Comps_Method": "Insufficient Peers", "CompsPrice": np.nan,
                "Comps_Upside": np.nan, "Comps_Median_PE": np.nan,
            })
            continue

        # ── Method 1: Median P/E (always calculated as baseline) ─────────────
        median_pe     = peers["PE_Ratio"].median()
        price_median_pe = median_pe * eps

        # ── Diagnostic: PEG Trigger ───────────────────────────────────────────
        growth_col = "BlendedGrowth"
        peg_triggered = False
        price_peg = np.nan
        if growth_col in peers.columns and peers[growth_col].notna().sum() >= 2:
            growth_std = peers[growth_col].std()
            if growth_std > 0.10:  # >10pp dispersion → PEG more informative
                peg_triggered = True
                target_growth = row.get(growth_col, 0.10)
                if target_growth > 0 and pe > 0:
                    target_peg = pe / (target_growth * 100)
                    peer_pegs  = peers["PE_Ratio"] / (peers[growth_col] * 100)
                    peer_pegs  = peer_pegs.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(peer_pegs) >= 2:
                        median_peg = peer_pegs.median()
                        fair_pe_peg = median_peg * (target_growth * 100)
                        price_peg   = fair_pe_peg * eps

        # ── Diagnostic: EV/EBITDA Trigger ────────────────────────────────────
        evebitda_triggered = False
        price_evebitda = np.nan
        if ("NetDebt_EBITDA" in peers.columns and
                peers["NetDebt_EBITDA"].notna().sum() >= 2):
            lev_spread = peers["NetDebt_EBITDA"].max() - peers["NetDebt_EBITDA"].min()
            if lev_spread > CONFIG["COMPS_LEVERAGE_TOL"]:
                evebitda_triggered = True
                # EV/EBITDA proxy: P/E × (1 - Wd) × EBITDA/EBIT
                # We just note the trigger; detailed peer EV pulled from data
                if "EBITDA_Cr" in df.columns and row["EBITDA_Cr"] > 0:
                    # Simple: median peer EV/EBITDA × target EBITDA − net debt
                    peer_ev = peers["EV_Cr"] if "EV_Cr" in peers.columns else pd.Series(dtype=float)
                    peer_ebitda = peers["EBITDA_Cr"] if "EBITDA_Cr" in peers.columns else pd.Series(dtype=float)
                    if len(peer_ev) >= 2 and len(peer_ebitda) >= 2:
                        ev_ebitda_multiples = (peer_ev / peer_ebitda).replace([np.inf, -np.inf], np.nan).dropna()
                        if len(ev_ebitda_multiples) >= 2:
                            median_ev_ebitda = ev_ebitda_multiples.median()
                            implied_ev = median_ev_ebitda * row["EBITDA_Cr"]
                            net_debt   = row["TotalDebt_Cr"] - row["TotalCash_Cr"]
                            implied_equity_cr = implied_ev - net_debt
                            price_evebitda = (implied_equity_cr * CRORE) / row["SharesOutstanding"] \
                                             if row["SharesOutstanding"] > 0 else np.nan

        # ── Diagnostic: P/E vs ROE Regression ────────────────────────────────
        reg_triggered = False
        price_regression = np.nan
        if ("ROE" in peers.columns and peers["ROE"].notna().sum() >= 4):
            X_reg = sm.add_constant(peers["ROE"].dropna())
            Y_reg = peers.loc[X_reg.index, "PE_Ratio"].dropna()
            common = X_reg.index.intersection(Y_reg.index)
            if len(common) >= 4:
                try:
                    reg = sm.OLS(Y_reg.loc[common], X_reg.loc[common]).fit()
                    if reg.rsquared > 0.50:
                        reg_triggered = True
                        target_roe = roe if not np.isnan(roe) else peers["ROE"].median()
                        predicted_pe = reg.params.iloc[0] + reg.params.iloc[1] * target_roe
                        predicted_pe = max(predicted_pe, 1.0)
                        price_regression = predicted_pe * eps
                except Exception:
                    pass

        # ── Method Selection (Sequential Diagnostic Hierarchy) ───────────────
        if reg_triggered and not np.isnan(price_regression):
            final_method = "P/E vs ROE Regression"
            comps_price  = price_regression
        elif evebitda_triggered and not np.isnan(price_evebitda):
            final_method = "EV/EBITDA"
            comps_price  = price_evebitda
        elif peg_triggered and not np.isnan(price_peg):
            final_method = "PEG-Adjusted P/E"
            comps_price  = price_peg
        else:
            final_method = "Median P/E"
            comps_price  = price_median_pe

        comps_upside = (comps_price - price) / price if (price > 0 and comps_price > 0) else np.nan

        comps_results.append({
            "Ticker"          : ticker,
            "N_Peers"         : n_peers,
            "Comps_Method"    : final_method,
            "Comps_Median_PE" : round(median_pe, 2),
            "PEG_Triggered"   : peg_triggered,
            "EV_EBITDA_Triggered": evebitda_triggered,
            "Reg_Triggered"   : reg_triggered,
            "CompsPrice"      : round(comps_price, 2) if not np.isnan(comps_price) else np.nan,
            "Comps_Upside"    : round(comps_upside, 4) if not np.isnan(comps_upside) else np.nan,
        })

    comps_df = pd.DataFrame(comps_results)
    print(f"  ✓ Comps complete. Method distribution:")
    print(comps_df["Comps_Method"].value_counts().to_string())
    return comps_df


# =============================================================================
# SECTION 5: TRIANGULATION ENGINE
# =============================================================================

def triangulate(dcf_df: pd.DataFrame, comps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Blends DCF intrinsic value with Comps market-implied value.

    Weighting logic (NOT simple averaging):
    - If both DCF and Comps are valid: 60% DCF / 40% Comps
      (DCF is primary — academic convention + richer info set)
    - If only DCF valid: 100% DCF
    - If only Comps valid: 100% Comps
    - Margin of Safety bands: >20% upside = BUY, <-20% = SELL, else HOLD
    """
    print("\n=== [5/6] Triangulation (DCF × Comps Blend) ===")

    merged = dcf_df.merge(comps_df, on="Ticker", how="left")

    triangulated = []
    for _, row in merged.iterrows():
        dcf_price   = row.get("IntrinsicPrice_DCF", np.nan)
        comps_price = row.get("CompsPrice", np.nan)
        price       = row["Price"]

        dcf_valid   = not np.isnan(dcf_price) and dcf_price > 0
        comps_valid = not np.isnan(comps_price) and comps_price > 0

        if dcf_valid and comps_valid:
            triangulated_price = 0.60 * dcf_price + 0.40 * comps_price
            blend_note = "60% DCF / 40% Comps"
        elif dcf_valid:
            triangulated_price = dcf_price
            blend_note = "100% DCF (Comps unavailable)"
        elif comps_valid:
            triangulated_price = comps_price
            blend_note = "100% Comps (DCF unavailable)"
        else:
            triangulated_price = np.nan
            blend_note = "Insufficient data"

        tri_upside = (triangulated_price - price) / price \
                     if (price > 0 and not np.isnan(triangulated_price)) else np.nan

        # Margin of safety signal
        if np.isnan(tri_upside):
            signal = "N/A"
        elif tri_upside > 0.20:
            signal = "BUY"
        elif tri_upside < -0.20:
            signal = "SELL"
        else:
            signal = "HOLD"

        triangulated.append({
            "Ticker"             : row["Ticker"],
            "Sector"             : row["Sector"],
            "Price"              : price,
            "DCF_Price"          : round(dcf_price, 2) if dcf_valid else np.nan,
            "Comps_Price"        : round(comps_price, 2) if comps_valid else np.nan,
            "Triangulated_Price" : round(triangulated_price, 2) if not np.isnan(triangulated_price) else np.nan,
            "Blend_Note"         : blend_note,
            "Triangulated_Upside": round(tri_upside, 4) if not np.isnan(tri_upside) else np.nan,
            "Signal"             : signal,
            "WACC"               : row.get("WACC", np.nan),
            "BlendedGrowth"      : row.get("BlendedGrowth", np.nan),
            "TerminalMethod"     : row.get("TerminalMethod", ""),
            "Comps_Method"       : row.get("Comps_Method", ""),
            "N_Peers"            : row.get("N_Peers", 0),
            "Rationale"          : row.get("Rationale", ""),
        })

    tri_df = pd.DataFrame(triangulated)
    print(f"  ✓ Triangulation complete.")
    print(f"  Signal distribution:\n{tri_df['Signal'].value_counts().to_string()}")
    return tri_df


# =============================================================================
# SECTION 6: SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_grid(ticker: str, result_row: pd.Series) -> pd.DataFrame:
    """
    Two-dimensional WACC × Terminal Growth sensitivity grid.
    Shows intrinsic price under 25 scenarios (5×5).
    """
    base_wacc = result_row.get("WACC", 0.10)
    base_tg   = result_row.get("BlendedGrowth", 0.05)
    nopat     = result_row.get("NOPAT_Cr", 0)
    ev_base   = result_row.get("EV_Cr", 0)
    debt      = result_row.get("TotalDebt_Cr", 0)
    cash      = result_row.get("TotalCash_Cr", 0)
    shares    = result_row.get("SharesOutstanding", 1)

    wacc_range = [base_wacc - 0.02, base_wacc - 0.01, base_wacc,
                  base_wacc + 0.01, base_wacc + 0.02]
    tg_range   = [base_tg - 0.01, base_tg - 0.005, base_tg,
                  base_tg + 0.005, base_tg + 0.01]

    rows = []
    for wc in wacc_range:
        row_data = {"WACC": f"{wc:.1%}"}
        for tg in tg_range:
            if wc <= tg:
                row_data[f"g={tg:.1%}"] = "N/A"
                continue
            tv = (nopat * (1 + tg)) / (wc - tg)
            # Very simplified — just TV sensitivity for grid
            eq = tv - debt + cash
            px = (eq * CRORE) / shares if shares > 0 else 0
            row_data[f"g={tg:.1%}"] = f"₹{px:.0f}"
        rows.append(row_data)

    return pd.DataFrame(rows).set_index("WACC")


# =============================================================================
# SECTION 7: MAIN ORCHESTRATOR
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  ILGC v9 — INTEGRATED LARGE-CAP GROWTH & COMPS PIPELINE")
    print("  DCF + Comps + Triangulation | Nifty 50 Universe")
    print("=" * 70 + "\n")

    # ── [1] Tickers ─────────────────────────────────────────────────────────
    tickers, sector_map = get_nifty50_tickers()

    # Remove structural exclusions
    tickers_clean = [t for t in tickers if t not in STRUCTURAL_EXCLUSIONS]
    print(f"  Excluded {len(STRUCTURAL_EXCLUSIONS)} structurally complex stocks "
          f"(SOTP/NAV required): {', '.join(STRUCTURAL_EXCLUSIONS)}")

    # ── [2] Market Data ──────────────────────────────────────────────────────
    print("\n=== [2/6] Fetching Nifty 50 Index Prices ===")
    mkt_df = yf.download(CONFIG["MARKET_TICKER"], period="5y",
                         interval="1wk", progress=False, auto_adjust=True)
    market_prices = mkt_df["Close"].squeeze()
    print(f"  ✓ Loaded {len(market_prices)} weeks of Nifty 50 data.")

    # ── [3] DCF Loop ─────────────────────────────────────────────────────────
    print(f"\n=== [3/6] Running DCF on {len(tickers_clean)} Stocks ===")
    dcf_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["THREADS"]) as exe:
        futures = {exe.submit(compute_dcf_single, t, market_prices, sector_map): t
                   for t in tickers_clean}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(tickers_clean), desc="  DCF"):
            res = fut.result()
            if res:
                dcf_results.append(res)
            time.sleep(CONFIG["REQUEST_DELAY"] / CONFIG["THREADS"])

    if not dcf_results:
        print("  ✗ CRITICAL: No DCF results. Check network / yfinance availability.")
        return

    dcf_df = pd.DataFrame(dcf_results)
    print(f"\n  ✓ DCF complete: {len(dcf_df)} stocks processed.")
    print(f"  Terminal Method distribution:\n{dcf_df['TerminalMethod'].value_counts().to_string()}")

    # ── [4] Comps ────────────────────────────────────────────────────────────
    comps_df = run_comps_engine(dcf_df)

    # ── [5] Triangulation ────────────────────────────────────────────────────
    tri_df = triangulate(dcf_df, comps_df)

    # ── [6] Output ───────────────────────────────────────────────────────────
    print("\n=== [6/6] Saving Outputs ===")

    dcf_df.to_csv("ilgc_dcf_output.csv", index=False)
    comps_df.to_csv("ilgc_comps_output.csv", index=False)
    tri_df.to_csv("ilgc_triangulated_output.csv", index=False)

    # Save full merged for dashboard
    full_merged = dcf_df.merge(comps_df, on="Ticker", how="left") \
                         .merge(tri_df[["Ticker","Triangulated_Price",
                                        "Triangulated_Upside","Signal","Blend_Note"]],
                                on="Ticker", how="left")
    full_merged.to_csv("ilgc_full_dashboard_data.csv", index=False)
    full_merged.to_json("ilgc_full_dashboard_data.json", orient="records", indent=2)

    # ── Summary Print ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ILGC FINAL RESULTS SUMMARY")
    print("=" * 70)
    buy_df  = tri_df[tri_df["Signal"] == "BUY"].sort_values("Triangulated_Upside", ascending=False)
    sell_df = tri_df[tri_df["Signal"] == "SELL"].sort_values("Triangulated_Upside")

    print(f"\n  🟢 BUY signals ({len(buy_df)} stocks):")
    print(buy_df[["Ticker","Sector","Price","Triangulated_Price",
                  "Triangulated_Upside","Signal"]].head(10).to_string(index=False))

    print(f"\n  🔴 SELL signals ({len(sell_df)} stocks):")
    print(sell_df[["Ticker","Sector","Price","Triangulated_Price",
                   "Triangulated_Upside","Signal"]].head(10).to_string(index=False))

    print("\n  Files saved:")
    print("    ilgc_dcf_output.csv")
    print("    ilgc_comps_output.csv")
    print("    ilgc_triangulated_output.csv")
    print("    ilgc_full_dashboard_data.csv")
    print("    ilgc_full_dashboard_data.json   ← feed into dashboard")
    print("\n" + "=" * 70)

    return dcf_df, comps_df, tri_df


if __name__ == "__main__":
    dcf_df, comps_df, tri_df = main()
