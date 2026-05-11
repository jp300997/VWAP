"""
=============================================================================
Part 3: Execution Application
Systematic Exploitation of Institutional Trading Signals
Jun Hyoung Park (541039718)

Defaults match the MT5 optimisation results shown in the capstone report:
  EMA=200, ADX_Period=12, ADX_Min=24, SD1=1.25, SD2=3.0
  Weekly VWAP buffer=0.04, Delta lookback=6, Delta min=0.15
  Risk=1.5%, ATR=20, SL_ATR=3.0, TP_ATR=2.5, Partial close=55% at ATR_1x

Trade history is stored in Supabase (Postgres) so it survives server
restarts, redeployments, and works on Streamlit Community Cloud.
The page auto-refreshes every 5 s via @st.fragment(run_every=...) with
zero flicker — only the data updates, not the full page.
=============================================================================
"""

import math
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from collections import deque
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# ── Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VWAP Institutional Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    div[data-testid="metric-container"] {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px 16px;
    }
    div[data-testid="metric-container"] label {
        color: #8b949e !important; font-size: 0.78rem;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #c9d1d9 !important; font-size: 1.3rem; font-weight: 700;
    }
    .signal-long  { background:#1a472a; border:1px solid #2ea043; color:#56d364;
                    padding:6px 14px; border-radius:6px; font-weight:700;
                    font-size:1.05rem; display:inline-block; }
    .signal-short { background:#4a1a1a; border:1px solid #f85149; color:#ff7b72;
                    padding:6px 14px; border-radius:6px; font-weight:700;
                    font-size:1.05rem; display:inline-block; }
    .signal-none  { background:#1c2128; border:1px solid #30363d; color:#8b949e;
                    padding:6px 14px; border-radius:6px; font-weight:700;
                    font-size:1.05rem; display:inline-block; }
    .open-trade   { background:#1a2a47; border:1px solid #58a6ff; color:#58a6ff;
                    padding:6px 14px; border-radius:6px; font-weight:700;
                    font-size:1.05rem; display:inline-block; }
    .section-title { color:#58a6ff; font-size:1rem; font-weight:600;
                     margin-bottom:4px; letter-spacing:.03em; }
    .alert-row  { border-left:3px solid #58a6ff; padding:4px 8px;
                  margin-bottom:4px; background:#161b22;
                  border-radius:0 4px 4px 0; font-size:0.82rem; }
    .alert-long  { border-left-color:#56d364; }
    .alert-short { border-left-color:#ff7b72; }
    /* Hide Streamlit top toolbar rerun button clutter */
    #MainMenu { visibility: hidden; }
    /* Custom metric cards — never truncate */
    .kpi-card {
        background:#161b22; border:1px solid #30363d; border-radius:8px;
        padding:10px 12px; min-width:0;
    }
    .kpi-label { color:#8b949e; font-size:0.72rem; font-weight:500;
                 letter-spacing:.04em; text-transform:uppercase;
                 white-space:nowrap; margin-bottom:4px; }
    .kpi-value { color:#c9d1d9; font-size:0.95rem; font-weight:700;
                 word-break:break-all; line-height:1.25; }
    .kpi-delta { font-size:0.72rem; font-weight:600; margin-top:3px; }
    .kpi-delta.up   { color:#56d364; }
    .kpi-delta.down { color:#ff7b72; }
    .kpi-delta.neutral { color:#8b949e; }
    /* Shrink native metric font globally too */
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size:0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── Constants
# ---------------------------------------------------------------------------
# Primary: Binance (may be blocked on some cloud hosts)
BINANCE_REST  = "https://api.binance.com/api/v3/klines"
BINANCE_TICK  = "https://api.binance.com/api/v3/ticker/price"
# Fallback: CoinGecko free API (no key required, works everywhere)
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_OHLC  = "https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
# Symbol → CoinGecko coin_id map
SYMBOL_TO_CG = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
}
BUFFER_SIZE   = 500
START_EQUITY  = 10_000.0
TICK_INTERVAL = 5                            # seconds between live price polls

# ---------------------------------------------------------------------------
# ── Supabase client  (credentials come from st.secrets on Streamlit Cloud,
#    or from a local .streamlit/secrets.toml file for local development)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# ---------------------------------------------------------------------------
# ── KPI card helper
# ---------------------------------------------------------------------------
def kpi(label: str, value: str, delta: str = "", delta_dir: str = "neutral") -> str:
    """Render a custom metric card that never truncates its value."""
    delta_html = (f'<div class="kpi-delta {delta_dir}">{delta}</div>'
                  if delta else "")
    return (f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{delta_html}</div>')


# ---------------------------------------------------------------------------
# ── Supabase persistence helpers
#
#  Schema (run once in Supabase SQL editor):
#
#  create table paper_state (
#      id          text primary key default 'singleton',
#      balance     float8 not null,
#      open_trade  jsonb,
#      last_sig_ts text,
#      alerts      jsonb
#  );
#  create table paper_closed (
#      id          bigserial primary key,
#      entry_time  text,
#      exit_time   text,
#      direction   text,
#      entry       float8,
#      sl          float8,
#      tp          float8,
#      exit_price  float8,
#      result      text,
#      risk_usd    float8,
#      lots        float8,
#      notional    float8,
#      pnl_usd     float8,
#      pnl_r       float8,
#      balance     float8
#  );
#  create table paper_equity (
#      id      bigserial primary key,
#      ts      text,
#      equity  float8
#  );
#  -- Allow anonymous reads/writes (RLS off for simplicity on a personal project)
#  alter table paper_state  disable row level security;
#  alter table paper_closed disable row level security;
#  alter table paper_equity disable row level security;
# ---------------------------------------------------------------------------

def _dt(val):
    """Parse ISO string to datetime, pass through if already datetime."""
    if val is None:
        return None
    if hasattr(val, "timestamp"):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


def _iso(val) -> str:
    """Convert datetime to ISO string for storage."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _load_persist() -> dict:
    """Load all state from Supabase. Returns default dict on any error."""
    default = {
        "pt_balance":      START_EQUITY,
        "pt_open_trade":   None,
        "pt_closed":       [],
        "pt_equity_curve": [{"time": datetime.now(timezone.utc), "equity": START_EQUITY}],
        "last_signal_ts":  None,
        "alerts":          [],
    }
    try:
        sb = get_supabase()

        # ── State row (balance, open trade, last signal, alerts) ─────────────
        res = sb.table("paper_state").select("*").eq("id", "singleton").execute()
        if res.data:
            row = res.data[0]
            default["pt_balance"] = float(row["balance"])
            default["last_signal_ts"] = _dt(row.get("last_sig_ts"))
            default["alerts"] = row.get("alerts") or []
            ot = row.get("open_trade")
            if ot:
                ot["entry_time"] = _dt(ot.get("entry_time"))
                default["pt_open_trade"] = ot

        # ── Closed trades ─────────────────────────────────────────────────────
        res2 = sb.table("paper_closed").select("*").order("id").execute()
        for t in (res2.data or []):
            t["entry_time"] = _dt(t.get("entry_time"))
            t["exit_time"]  = _dt(t.get("exit_time"))
            default["pt_closed"].append(t)

        # ── Equity curve ──────────────────────────────────────────────────────
        res3 = sb.table("paper_equity").select("*").order("id").execute()
        if res3.data:
            default["pt_equity_curve"] = [
                {"time": _dt(p["ts"]), "equity": float(p["equity"])}
                for p in res3.data
            ]

    except Exception as e:
        st.warning(f"Supabase load error (using defaults): {e}")

    return default


def _save_state() -> None:
    """Upsert the singleton state row (balance, open trade, alerts)."""
    try:
        sb  = get_supabase()
        ot  = st.session_state.get("pt_open_trade")
        ot_serial = None
        if ot:
            ot_serial = dict(ot)
            ot_serial["entry_time"] = _iso(ot_serial.get("entry_time"))

        sb.table("paper_state").upsert({
            "id":          "singleton",
            "balance":     st.session_state["pt_balance"],
            "open_trade":  ot_serial,
            "last_sig_ts": _iso(st.session_state.get("last_signal_ts")),
            "alerts":      list(st.session_state["alerts"]),
        }).execute()
    except Exception as e:
        st.warning(f"Supabase save error: {e}")


def _append_closed_trade(trade_dict: dict) -> None:
    """Insert one closed trade row."""
    try:
        sb  = get_supabase()
        row = {k: v for k, v in trade_dict.items()
               if k not in ("entry_time", "exit_time")}
        row["entry_time"] = _iso(trade_dict.get("entry_time"))
        row["exit_time"]  = _iso(trade_dict.get("exit_time"))
        sb.table("paper_closed").insert(row).execute()
    except Exception as e:
        st.warning(f"Supabase trade insert error: {e}")


def _append_equity_point(ts, equity: float) -> None:
    """Insert one equity curve point."""
    try:
        sb = get_supabase()
        sb.table("paper_equity").insert({
            "ts":     _iso(ts),
            "equity": equity,
        }).execute()
    except Exception as e:
        st.warning(f"Supabase equity insert error: {e}")


def _reset_supabase() -> None:
    """Delete all rows from all three tables (called on account reset)."""
    try:
        sb = get_supabase()
        sb.table("paper_state").delete().eq("id", "singleton").execute()
        sb.table("paper_closed").delete().neq("id", 0).execute()
        sb.table("paper_equity").delete().neq("id", 0).execute()
    except Exception as e:
        st.warning(f"Supabase reset error: {e}")

# ---------------------------------------------------------------------------
# ── Session state bootstrap  (loads from Supabase on first run per session)
# ---------------------------------------------------------------------------
if "pt_loaded" not in st.session_state:
    disk = _load_persist()
    st.session_state["pt_balance"]      = disk["pt_balance"]
    st.session_state["pt_open_trade"]   = disk["pt_open_trade"]
    st.session_state["pt_closed"]       = disk["pt_closed"]
    st.session_state["pt_equity_curve"] = disk["pt_equity_curve"]
    st.session_state["last_signal_ts"]  = disk["last_signal_ts"]
    st.session_state["alerts"]          = deque(disk.get("alerts", []), maxlen=50)
    st.session_state["pt_loaded"]       = True

# ---------------------------------------------------------------------------
# ── Sidebar  ── optimised defaults from MT5 backtest
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Strategy Parameters")
st.sidebar.caption("Defaults = optimised MT5 backtest values")

symbol   = st.sidebar.selectbox("Asset", ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"], index=0)
interval = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1h"], index=2)

st.sidebar.markdown("---")
st.sidebar.markdown("**Trend & Momentum**")
ema_period = st.sidebar.slider("EMA Period",      100, 300, 200, step=50)
adx_period = st.sidebar.slider("ADX Period",       10,  20,  12, step=2)
adx_min    = st.sidebar.slider("ADX Minimum",      18,  32,  24, step=2)

st.sidebar.markdown("**VWAP SD Band Settings**")
sd1_multi  = st.sidebar.slider("SD1 (Inner) Multiplier", 0.50, 2.00, 1.25, step=0.25)
sd2_multi  = st.sidebar.slider("SD2 (Outer) Multiplier", 1.50, 4.00, 3.00, step=0.50)

st.sidebar.markdown("**Weekly VWAP Confluence**")
use_weekly = st.sidebar.checkbox("Use Weekly VWAP", value=True)
weekly_buf = st.sidebar.slider("Weekly VWAP Buffer %", 0.02, 0.12, 0.04, step=0.02)

st.sidebar.markdown("**Tier 2: Volume Profile**")
use_vol_profile = st.sidebar.checkbox("Use Volume Profile", value=True)
vp_bins         = st.sidebar.slider("Price Bins",       10, 40, 25, step=5)
va_percent      = st.sidebar.slider("Value Area %",     60, 80, 75, step=5)
poc_buffer      = st.sidebar.slider("Near-POC Buffer %", 0.10, 0.30, 0.25, step=0.05)

st.sidebar.markdown("**Tier 2: Delta Divergence**")
use_delta = st.sidebar.checkbox("Use Delta Filter", value=True)
delta_lb  = st.sidebar.slider("Delta Lookback Bars",  2,  6,  6, step=1)
delta_min_str = st.sidebar.slider("Delta Min Strength", 0.05, 0.30, 0.15, step=0.05)

st.sidebar.markdown("**Risk & Volatility**")
risk_pct   = st.sidebar.slider("Risk % per Trade",  1.0,  2.5, 1.5, step=0.5)
atr_period = st.sidebar.slider("ATR Period",         10,  22,  20, step=2)
sl_atr     = st.sidebar.slider("SL × ATR",          1.5,  4.0, 3.0, step=0.5)
tp_atr     = st.sidebar.slider("TP × ATR",          2.0,  7.0, 2.5, step=0.5)
max_spread = st.sidebar.slider("Max Spread %",      0.01, 1.0, 0.15, step=0.01)

st.sidebar.markdown("**Partial Close**")
use_partial   = st.sidebar.checkbox("Use Partial Close", value=True)
partial_pct   = st.sidebar.slider("Partial Close %",   25, 75, 55, step=5)
# Partial target: 1 = ATR_1x (optimised value)
partial_target = st.sidebar.selectbox(
    "First Target", ["BB Mid (0)", "1×ATR (1)", "VWAP (2)"], index=1)
partial_target_idx = int(partial_target.split("(")[1].rstrip(")"))

st.sidebar.markdown("---")
if st.sidebar.button("🗑️  Reset Paper Account"):
    for key in ["pt_balance","pt_open_trade","pt_closed",
                "pt_equity_curve","alerts","last_signal_ts","pt_loaded"]:
        if key in st.session_state:
            del st.session_state[key]
    _reset_supabase()
    st.rerun()

# ---------------------------------------------------------------------------
# ── Data ingestion
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def _candles_from_binance(sym: str, tf: str, limit: int) -> pd.DataFrame:
    """Attempt to fetch OHLCV from Binance REST."""
    params = {"symbol": sym, "interval": tf, "limit": limit}
    resp = requests.get(BINANCE_REST, params=params, timeout=8)
    resp.raise_for_status()
    raw  = resp.json()
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df.reset_index(drop=True)


def _candles_from_coingecko(sym: str, tf: str, limit: int) -> pd.DataFrame:
    """
    Fallback: CoinGecko /ohlc endpoint.
    Returns daily candles (days=14 or 30) resampled to approximate the
    requested timeframe. CoinGecko free tier only offers 1/7/14/30 day windows.
    Sufficient for signal generation on 15m/1h timeframes.
    """
    coin_id = SYMBOL_TO_CG.get(sym)
    if not coin_id:
        return pd.DataFrame()

    # Map timeframe to days window
    days = 1 if tf in ("1m", "5m", "15m") else 7

    url  = COINGECKO_OHLC.format(coin_id=coin_id)
    resp = requests.get(url, params={"vs_currency": "usd", "days": days},
                        timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame()

    # CoinGecko returns [[timestamp_ms, open, high, low, close], ...]
    df = pd.DataFrame(raw, columns=["open_time","open","high","low","close"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    df["volume"] = 0.0   # CoinGecko OHLC doesn't include volume
    df["trades"] = 0
    df = df.sort_values("open_time").reset_index(drop=True)

    # Keep last `limit` rows
    return df.tail(limit).reset_index(drop=True)


@st.cache_data(ttl=60)
def fetch_candles(sym: str, tf: str, limit: int = BUFFER_SIZE) -> pd.DataFrame:
    """
    Fetch OHLCV candles. Tries Binance first; falls back to CoinGecko
    if Binance is unreachable (e.g. on Streamlit Community Cloud).
    """
    df = pd.DataFrame()

    # ── Try Binance ──────────────────────────────────────────────────────────
    try:
        df = _candles_from_binance(sym, tf, limit)
    except Exception:
        pass

    # ── Fallback to CoinGecko ────────────────────────────────────────────────
    if df.empty:
        try:
            df = _candles_from_coingecko(sym, tf, limit)
        except Exception as e:
            st.error(f"Could not fetch candle data from Binance or CoinGecko: {e}")
            return pd.DataFrame()

    if df.empty:
        return df

    # §4.1.3 OHLC validity checks
    price_range = df["high"] - df["low"]
    high_viol   = df["high"] < df[["open","close"]].max(axis=1)
    low_viol    = df["low"]  > df[["open","close"]].min(axis=1)
    for mask, col, func in [(high_viol,"high","max"),(low_viol,"low","min")]:
        ref       = df[["open","close"]].agg(func, axis=1)
        deviation = (ref - df[col]).abs()
        minor = mask & (deviation <= 0.01 * price_range.clip(lower=1e-8))
        major = mask & ~minor
        df.loc[minor, col]      = ref[minor]
        df.loc[major, "volume"] = 0.0

    return df.reset_index(drop=True)


def fetch_live_price(sym: str) -> float:
    """
    Fetch latest price. Tries Binance ticker first (fast),
    falls back to CoinGecko simple price API if blocked.
    """
    # ── Binance ──────────────────────────────────────────────────────────────
    try:
        r = requests.get(BINANCE_TICK, params={"symbol": sym}, timeout=4)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        pass

    # ── CoinGecko fallback ───────────────────────────────────────────────────
    try:
        coin_id = SYMBOL_TO_CG.get(sym)
        if coin_id:
            r = requests.get(
                COINGECKO_PRICE,
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=6,
            )
            r.raise_for_status()
            return float(r.json()[coin_id]["usd"])
    except Exception:
        pass

    return 0.0

# ---------------------------------------------------------------------------
# ── Indicators
# ---------------------------------------------------------------------------
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    up       = df["high"].diff()
    down     = -df["low"].diff()
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr      = compute_atr(df, period)
    plus_di  = (100 * pd.Series(plus_dm, index=df.index)
                .ewm(span=period, adjust=False).mean()
                / atr.clip(lower=1e-8))
    minus_di = (100 * pd.Series(minus_dm, index=df.index)
                .ewm(span=period, adjust=False).mean()
                / atr.clip(lower=1e-8))
    dx = (100 * (plus_di - minus_di).abs()
          / (plus_di + minus_di).clip(lower=1e-8))
    return dx.ewm(span=period, adjust=False).mean()

def compute_daily_vwap(df: pd.DataFrame) -> pd.DataFrame:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day_key = df["open_time"].dt.floor("D")
    cumvol  = df.groupby(day_key)["volume"].cumsum()
    cum_pv  = (typical * df["volume"]).groupby(day_key).cumsum()
    cum_pv2 = (typical**2 * df["volume"]).groupby(day_key).cumsum()
    vwap    = cum_pv / cumvol.clip(lower=1e-8)
    var     = (cum_pv2 / cumvol.clip(lower=1e-8)) - vwap**2
    sd      = var.clip(lower=0).apply(math.sqrt)
    return pd.DataFrame({
        "vwap":   vwap, "sd": sd,
        "upper1": vwap + sd * sd1_multi,
        "lower1": vwap - sd * sd1_multi,
        "upper2": vwap + sd * sd2_multi,
        "lower2": vwap - sd * sd2_multi,
    })

def compute_weekly_vwap(df: pd.DataFrame) -> pd.Series:
    typical  = (df["high"] + df["low"] + df["close"]) / 3.0
    iso      = df["open_time"].dt.isocalendar()
    week_key = (iso["year"].astype(str) + "-W"
                + iso["week"].astype(str).str.zfill(2))
    cumvol   = df.groupby(week_key)["volume"].cumsum()
    cum_pv   = (typical * df["volume"]).groupby(week_key).cumsum()
    return (cum_pv / cumvol.clip(lower=1e-8)).rename("weekly_vwap")

def compute_volume_profile(df: pd.DataFrame):
    """
    Session volume profile: POC, VAH, VAL.
    Mirrors CalculateVolumeProfile() in the MQL5 EA.
    Returns (poc, vah, val) floats, or (0,0,0) if insufficient data.
    """
    today = df["open_time"].iloc[-1].floor("D")
    day_df = df[df["open_time"] >= today].copy()
    if len(day_df) < 3:
        return 0.0, 0.0, 0.0

    day_high = day_df["high"].max()
    day_low  = day_df["low"].min()
    if day_high <= day_low:
        return 0.0, 0.0, 0.0

    bin_size = (day_high - day_low) / vp_bins
    bins     = np.zeros(vp_bins)

    for _, row in day_df.iterrows():
        bar_range = row["high"] - row["low"]
        if bar_range <= 0:
            continue
        for b in range(vp_bins):
            bin_lo  = day_low + b * bin_size
            bin_hi  = bin_lo + bin_size
            overlap = min(row["high"], bin_hi) - max(row["low"], bin_lo)
            if overlap > 0:
                bins[b] += row["volume"] * (overlap / bar_range)

    total_vol = bins.sum()
    if total_vol <= 0:
        return 0.0, 0.0, 0.0

    poc_bin = int(np.argmax(bins))
    poc     = day_low + (poc_bin + 0.5) * bin_size

    target   = total_vol * (va_percent / 100.0)
    captured = bins[poc_bin]
    va_lo, va_hi = poc_bin, poc_bin

    while captured < target and (va_lo > 0 or va_hi < vp_bins - 1):
        add_below = bins[va_lo - 1] if va_lo > 0 else 0
        add_above = bins[va_hi + 1] if va_hi < vp_bins - 1 else 0
        if add_above >= add_below and va_hi < vp_bins - 1:
            va_hi += 1; captured += bins[va_hi]
        elif va_lo > 0:
            va_lo -= 1; captured += bins[va_lo]
        elif va_hi < vp_bins - 1:
            va_hi += 1; captured += bins[va_hi]
        else:
            break

    val = day_low + va_lo * bin_size
    vah = day_low + (va_hi + 1) * bin_size
    return poc, vah, val

def compute_delta_ratio(df: pd.DataFrame, lookback: int) -> pd.Series:
    rng   = (df["high"] - df["low"]).clip(lower=1e-8)
    delta = (df["close"] - df["open"]) / rng
    return delta.rolling(lookback).mean().rename("delta_ratio")

def compute_zscore(df: pd.DataFrame, vwap_df: pd.DataFrame) -> pd.Series:
    sd = vwap_df["sd"].clip(lower=1e-8)
    return ((df["close"] - vwap_df["vwap"]) / sd).rename("zscore")

def compute_bb_mid(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["close"].rolling(period).mean()

# ---------------------------------------------------------------------------
# ── Signal generation  (full Tier 1 + Tier 2 logic matching MQL5 EA)
# ---------------------------------------------------------------------------
def generate_signals(df, vwap_df, ema_ser, adx_ser, delta_s, w_vwap,
                     poc, vah, val):
    spread_pct = (df["high"] - df["low"]) / df["close"].clip(lower=1e-8) * 100
    signals    = pd.Series("", index=df.index)

    for i in range(max(ema_period, 50), len(df) - 1):
        c = df["close"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]

        vwap_v  = vwap_df["vwap"].iloc[i]
        upper1  = vwap_df["upper1"].iloc[i]
        lower1  = vwap_df["lower1"].iloc[i]
        upper2  = vwap_df["upper2"].iloc[i]
        lower2  = vwap_df["lower2"].iloc[i]
        ema_v   = ema_ser.iloc[i]
        adx_v   = adx_ser.iloc[i]
        dlt     = delta_s.iloc[i]

        if any(pd.isna(x) for x in [vwap_v, ema_v, adx_v, dlt]):
            continue
        if spread_pct.iloc[i] > max_spread:
            continue

        # Tier 1: trend + momentum
        bull_trend  = (c > ema_v) and (adx_v > adx_min)
        bear_trend  = (c < ema_v) and (adx_v > adx_min)
        long_touch  = (l <= upper1) and (l >= vwap_v)
        long_close  = c > vwap_v
        short_touch = (h >= lower1) and (h <= vwap_v)
        short_close = c < vwap_v
        over_up     = c > upper2
        over_down   = c < lower2

        # Tier 1: weekly VWAP confluence
        if use_weekly and w_vwap is not None:
            wv          = w_vwap.iloc[i]
            wbuf        = wv * (weekly_buf / 100.0)
            weekly_bull = c > wv + wbuf
            weekly_bear = c < wv - wbuf
        else:
            weekly_bull = weekly_bear = True

        # Tier 2A: volume profile filter
        if use_vol_profile and poc > 0:
            vp_long_ok  = c >= val   # above value area low
            vp_short_ok = c <= vah   # below value area high
        else:
            vp_long_ok = vp_short_ok = True

        # Tier 2B: delta divergence filter
        delta_long  = (not use_delta) or (dlt >= delta_min_str)
        delta_short = (not use_delta) or (dlt <= -delta_min_str)

        if (bull_trend and weekly_bull and long_touch and long_close
                and not over_up and vp_long_ok and delta_long):
            signals.iloc[i] = "LONG"
        elif (bear_trend and weekly_bear and short_touch and short_close
              and not over_down and vp_short_ok and delta_short):
            signals.iloc[i] = "SHORT"

    return signals

# ---------------------------------------------------------------------------
# ── Paper trading engine
# ---------------------------------------------------------------------------
def _persist_state():
    """Persist current session state to Supabase (state row only)."""
    _save_state()


def _close_trade(trade: dict, exit_price: float,
                 result: str, exit_time) -> None:
    direction  = trade["direction"]
    raw_pnl    = ((exit_price - trade["entry"]) if direction == "LONG"
                  else (trade["entry"] - exit_price))
    dollar_pnl = raw_pnl * trade["lots"]
    pnl_r      = raw_pnl / trade["sl_dist"]

    st.session_state["pt_balance"] += dollar_pnl
    notional = trade["entry"] * trade["lots"]
    st.session_state["pt_closed"].append({
        "entry_time":  exit_time,   # use exit_time as display time
        "exit_time":   exit_time,
        "direction":   direction,
        "entry":       trade["entry"],
        "sl":          trade["sl"],
        "tp":          trade["tp"],
        "exit_price":  exit_price,
        "result":      result,
        "risk_usd":    round(trade["risk_usd"], 2),
        "lots":        round(trade["lots"], 6),
        "notional":    round(notional, 2),
        "pnl_usd":     round(dollar_pnl, 2),
        "pnl_r":       round(pnl_r, 3),
        "balance":     round(st.session_state["pt_balance"], 2),
    })
    st.session_state["pt_equity_curve"].append({
        "time":   exit_time,
        "equity": st.session_state["pt_balance"],
    })
    st.session_state["pt_open_trade"] = None
    _persist_state()


def check_open_trade_vs_price(live_price: float) -> bool:
    """
    Called every tick. Checks if live price has crossed SL or TP.
    Returns True if trade was closed.
    """
    ot = st.session_state.get("pt_open_trade")
    if ot is None:
        return False
    now = datetime.now(timezone.utc)
    if ot["direction"] == "LONG":
        if live_price <= ot["sl"]:
            _close_trade(ot, ot["sl"], "LOSS", now); return True
        if live_price >= ot["tp"]:
            _close_trade(ot, ot["tp"], "WIN",  now); return True
    else:
        if live_price >= ot["sl"]:
            _close_trade(ot, ot["sl"], "LOSS", now); return True
        if live_price <= ot["tp"]:
            _close_trade(ot, ot["tp"], "WIN",  now); return True
    return False


def paper_trade_engine(df: pd.DataFrame, signals: pd.Series,
                       atr_ser: pd.Series, vwap_df: pd.DataFrame,
                       bb_mid: pd.Series) -> None:
    """
    Opens a new paper trade when a new signal bar appears.
    SL/TP checking against historical candles is done here on load;
    live tick checking is done in check_open_trade_vs_price().
    """
    ot = st.session_state["pt_open_trade"]

    # Check existing open trade against historical bars (catch missed hits)
    if ot is not None:
        for j in range(ot["entry_bar_idx"] + 1, len(df)):
            hi   = df["high"].iloc[j]
            lo   = df["low"].iloc[j]
            ts_j = df["open_time"].iloc[j]
            if ot["direction"] == "LONG":
                if lo <= ot["sl"]:
                    _close_trade(ot, ot["sl"], "LOSS", ts_j); ot = None; break
                if hi >= ot["tp"]:
                    _close_trade(ot, ot["tp"], "WIN",  ts_j); ot = None; break
            else:
                if hi >= ot["sl"]:
                    _close_trade(ot, ot["sl"], "LOSS", ts_j); ot = None; break
                if lo <= ot["tp"]:
                    _close_trade(ot, ot["tp"], "WIN",  ts_j); ot = None; break

    # Open a new trade on fresh signal
    if ot is None:
        sig_bar = len(df) - 2
        sig     = signals.iloc[sig_bar]
        bar_ts  = df["open_time"].iloc[sig_bar]
        last_ts = st.session_state["last_signal_ts"]

        if sig != "" and (last_ts is None or
                          bar_ts.isoformat() != (last_ts.isoformat()
                                                  if hasattr(last_ts, "isoformat")
                                                  else str(last_ts))):
            entry   = df["close"].iloc[sig_bar]
            atr_val = atr_ser.iloc[sig_bar]
            sl_dist = atr_val * sl_atr
            tp_dist = atr_val * tp_atr
            sl = entry - sl_dist if sig == "LONG" else entry + sl_dist
            tp = entry + tp_dist if sig == "LONG" else entry - tp_dist

            # First partial target price
            if partial_target_idx == 0:
                first_target = bb_mid.iloc[sig_bar]
            elif partial_target_idx == 1:
                first_target = (entry + atr_val if sig == "LONG"
                                else entry - atr_val)
            else:
                first_target = vwap_df["vwap"].iloc[sig_bar]

            balance  = st.session_state["pt_balance"]
            risk_usd = balance * (risk_pct / 100.0)
            lots     = risk_usd / sl_dist if sl_dist > 0 else 0

            new_trade = {
                "direction":      sig,
                "entry":          entry,
                "sl":             sl,
                "tp":             tp,
                "sl_dist":        sl_dist,
                "lots":           lots,
                "risk_usd":       risk_usd,
                "entry_time":     bar_ts,
                "entry_bar_idx":  sig_bar,
                "first_target":   first_target,
                "partial_done":   False,
            }
            st.session_state["pt_open_trade"]  = new_trade
            st.session_state["last_signal_ts"] = bar_ts
            st.session_state["alerts"].appendleft({
                "time":      bar_ts.strftime("%H:%M UTC"),
                "direction": sig,
                "price":     entry,
                "atr":       atr_val,
            })
            _persist_state()


def unrealised_pnl(price: float) -> float:
    ot = st.session_state.get("pt_open_trade")
    if ot is None:
        return 0.0
    raw = ((price - ot["entry"]) if ot["direction"] == "LONG"
           else (ot["entry"] - price))
    return raw * ot["lots"]

# ===========================================================================
# ── STATIC HEADER  (renders once, never flickers)
# ===========================================================================
st.markdown("# 📈 VWAP Institutional Signal Dashboard")
st.markdown(f"**{symbol}** · {interval} · Binance REST · Optimised parameters active")
st.markdown("---")


# ===========================================================================
# ── LIVE FRAGMENT
#    run_every=TICK_INTERVAL causes Streamlit to silently re-run only the
#    contents of this function on a timer — the static header above and the
#    sidebar never re-render, so there is zero flicker.
#    fetch_candles() is cached for 60 s so the heavy indicator compute
#    only fires once per minute; live price uses the fast ticker endpoint.
# ===========================================================================
@st.fragment(run_every=TICK_INTERVAL)
def live_dashboard():
    # ── Candles + indicators ─────────────────────────────────────────────────
    df = fetch_candles(symbol, interval)
    if df.empty:
        st.error("Could not fetch candle data from Binance. Check connection.")
        return

    vwap_df = compute_daily_vwap(df)
    w_vwap  = compute_weekly_vwap(df)
    ema_ser = compute_ema(df["close"], ema_period)
    atr_ser = compute_atr(df, atr_period)
    adx_ser = compute_adx(df, adx_period)
    delta_s = compute_delta_ratio(df, delta_lb)
    zscore  = compute_zscore(df, vwap_df)
    bb_mid  = compute_bb_mid(df, 20)
    poc, vah, val = compute_volume_profile(df)
    signals = generate_signals(df, vwap_df, ema_ser, adx_ser, delta_s,
                               w_vwap, poc, vah, val)

    # ── Paper trading engine ─────────────────────────────────────────────────
    paper_trade_engine(df, signals, atr_ser, vwap_df, bb_mid)

    # ── Live price ───────────────────────────────────────────────────────────
    live_price = fetch_live_price(symbol)
    if live_price <= 0:
        live_price = df["close"].iloc[-1]
    check_open_trade_vs_price(live_price)

    # ── Derived account values ───────────────────────────────────────────────
    unreal        = unrealised_pnl(live_price)
    balance       = st.session_state["pt_balance"]
    equity        = balance + unreal
    closed_trades = st.session_state["pt_closed"]
    open_trade    = st.session_state["pt_open_trade"]
    total_pnl     = equity - START_EQUITY
    total_pnl_pct = total_pnl / START_EQUITY * 100

    # ── Row 1: Market metrics ────────────────────────────────────────────────
    prev      = df.iloc[-2]
    price_chg = live_price - prev["close"]
    price_pct = price_chg / prev["close"] * 100

    _wv      = w_vwap.iloc[-1]
    _ema     = ema_ser.iloc[-1]
    _adx     = adx_ser.iloc[-1]
    _wv_dir  = "up"      if live_price > _wv  else "down"
    _ema_dir = "up"      if live_price > _ema else "down"
    _wv_lbl  = "Above ✅" if live_price > _wv  else "Below ❌"
    _ema_lbl = "Above ✅" if live_price > _ema else "Below ❌"
    _px_dir  = "up"      if price_chg >= 0    else "down"
    _adx_lbl = "▲ Trending" if _adx > adx_min else "▼ Ranging"
    _adx_dir = "up"      if _adx > adx_min    else "neutral"

    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-bottom:4px;">'
        + kpi("Live Price",        f"${live_price:,.2f}",
              f"{price_chg:+.2f} ({price_pct:+.2f}%)", _px_dir)
        + kpi("Daily VWAP",        f"${vwap_df['vwap'].iloc[-1]:,.2f}")
        + kpi("Weekly VWAP",       f"${_wv:,.2f}",  _wv_lbl,  _wv_dir)
        + kpi(f"EMA {ema_period}", f"${_ema:,.2f}", _ema_lbl, _ema_dir)
        + kpi("VWAP Z-Score",      f"{zscore.iloc[-1]:.2f}")
        + kpi("ADX",               f"{_adx:.1f}", _adx_lbl, _adx_dir)
        + kpi("Delta Ratio",       f"{delta_s.iloc[-1]:+.3f}")
        + kpi("ATR",               f"${atr_ser.iloc[-1]:,.2f}")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Row 2: Paper account ─────────────────────────────────────────────────
    st.markdown("")
    st.markdown('<p class="section-title">💰 PAPER TRADING ACCOUNT</p>',
                unsafe_allow_html=True)
    _pnl_dir = "up" if total_pnl >= 0 else "down"
    _unr_dir = "up" if unreal   >= 0 else "down"
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;">'
        + kpi("Start",         f"${START_EQUITY:,.2f}")
        + kpi("Cash Balance",  f"${balance:,.2f}")
        + kpi("Unrealised",    f"${unreal:+,.2f}", "", _unr_dir)
        + kpi("Equity",        f"${equity:,.2f}")
        + kpi("Net P&L",       f"${total_pnl:+,.2f}",
              f"{total_pnl_pct:+.2f}%", _pnl_dir)
        + kpi("Closed Trades", str(len(closed_trades)))
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Performance stats ────────────────────────────────────────────────────
    st.markdown("")
    st.markdown('<p class="section-title" style="margin-top:10px;">📈 PERFORMANCE STATS</p>',
                unsafe_allow_html=True)
    if closed_trades:
        _ct   = pd.DataFrame(closed_trades)
        _wins = _ct[_ct["result"] == "WIN"]
        _loss = _ct[_ct["result"] == "LOSS"]
        _wr   = len(_wins) / len(_ct) * 100
        _pf   = (_wins["pnl_usd"].sum() / abs(_loss["pnl_usd"].sum())
                 if len(_loss) > 0 and _loss["pnl_usd"].sum() != 0
                 else float("inf"))
        _eq_vals = [p["equity"] for p in st.session_state["pt_equity_curve"]]
        if open_trade is not None:
            _eq_vals = _eq_vals + [equity]
        _eq_arr  = np.array(_eq_vals)
        _peak    = np.maximum.accumulate(_eq_arr)
        _dd      = (_eq_arr - _peak) / _peak * 100
        _max_dd  = float(_dd.min())
        _returns = _ct["pnl_r"].values
        _sharpe  = (float(np.mean(_returns) / np.std(_returns) * np.sqrt(252))
                    if len(_returns) > 1 and np.std(_returns) > 0 else 0.0)
        _pf_str  = f"{_pf:.2f}" if _pf != float("inf") else "∞"
        _sh_lbl  = "▲ Good" if _sharpe > 1 else ("▲ OK" if _sharpe > 0 else "▼ Poor")
        _sh_dir  = "up"  if _sharpe > 1 else ("neutral" if _sharpe > 0 else "down")
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:4px;">'
            + kpi("Win Rate",      f"{_wr:.1f}%",
                  f"{len(_wins)}W / {len(_loss)}L", "up" if _wr >= 50 else "down")
            + kpi("Profit Factor", _pf_str)
            + kpi("Sharpe Ratio",  f"{_sharpe:.2f}", _sh_lbl, _sh_dir)
            + kpi("Max Drawdown",  f"{_max_dd:.2f}%",
                  "", "down" if _max_dd < -5 else "neutral")
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Stats will appear after the first trade closes.")

    # ── Volume profile ───────────────────────────────────────────────────────
    if use_vol_profile and poc > 0:
        st.markdown("")
        st.markdown('<p class="section-title" style="margin-top:8px;">📦 SESSION VOLUME PROFILE</p>',
                    unsafe_allow_html=True)
        near_poc = abs(live_price - poc) <= poc * (poc_buffer / 100)
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">'
            + kpi("POC",      f"${poc:,.2f}")
            + kpi("VAH",      f"${vah:,.2f}")
            + kpi("VAL",      f"${val:,.2f}")
            + kpi("Near POC?","✅ YES" if near_poc else "— no")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Row 3: Signal + open trade + checklist ───────────────────────────────
    sig_col, trade_col, cond_col = st.columns([1, 1, 2])
    sig_now = signals.iloc[-2]

    with sig_col:
        st.markdown('<p class="section-title">LATEST SIGNAL</p>',
                    unsafe_allow_html=True)
        if sig_now == "LONG":
            st.markdown('<span class="signal-long">▲ LONG SIGNAL</span>',
                        unsafe_allow_html=True)
        elif sig_now == "SHORT":
            st.markdown('<span class="signal-short">▼ SHORT SIGNAL</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="signal-none">NO SIGNAL</span>',
                        unsafe_allow_html=True)
            st.caption("Awaiting VWAP SD-band touch + filter confluence")

    with trade_col:
        st.markdown('<p class="section-title">OPEN PAPER TRADE</p>',
                    unsafe_allow_html=True)
        if open_trade is not None:
            ot    = open_trade
            arrow = "▲" if ot["direction"] == "LONG" else "▼"
            st.markdown(
                f'<span class="open-trade">{arrow} {ot["direction"]} ACTIVE</span>',
                unsafe_allow_html=True)
            st.caption(
                f"Entry ${ot['entry']:,.2f}  |  "
                f"SL ${ot['sl']:,.2f}  |  TP ${ot['tp']:,.2f}")
            pnl_r  = ((live_price - ot["entry"]) / ot["sl_dist"]
                      if ot["direction"] == "LONG"
                      else (ot["entry"] - live_price) / ot["sl_dist"])
            colour = "#56d364" if unreal >= 0 else "#ff7b72"
            st.markdown(
                f"<span style='color:{colour}; font-weight:700;'>"
                f"Float: ${unreal:+,.2f}  ({pnl_r:+.2f}R)</span>",
                unsafe_allow_html=True)
            if use_partial:
                st.caption(
                    f"Partial target ({partial_pct}%): "
                    f"${ot.get('first_target', 0):,.2f}")
        else:
            st.markdown('<span class="signal-none">No open trade</span>',
                        unsafe_allow_html=True)
            st.caption("Next signal will open a simulated position")

    with cond_col:
        st.markdown(
            '<p class="section-title">CONDITION CHECKLIST (last closed bar)</p>',
            unsafe_allow_html=True)
        ci  = -2
        c_p = df["close"].iloc[ci]
        h_p = df["high"].iloc[ci]
        l_p = df["low"].iloc[ci]
        v   = vwap_df["vwap"].iloc[ci]
        u1  = vwap_df["upper1"].iloc[ci]
        l1  = vwap_df["lower1"].iloc[ci]
        ema = ema_ser.iloc[ci]
        adx = adx_ser.iloc[ci]
        dlt = delta_s.iloc[ci]
        wv  = w_vwap.iloc[ci]

        def chk(cond): return "✅" if cond else "❌"

        bull_t = (c_p > ema) and (adx > adx_min)
        bear_t = (c_p < ema) and (adx > adx_min)
        ltouch = (l_p <= u1) and (l_p >= v)
        stouch = (h_p >= l1) and (h_p <= v)
        wbull  = c_p > wv * (1 + weekly_buf/100) if use_weekly else True
        wbear  = c_p < wv * (1 - weekly_buf/100) if use_weekly else True
        vplong = c_p >= val if use_vol_profile and val > 0 else True
        vpshrt = c_p <= vah if use_vol_profile and vah > 0 else True

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.markdown(f"{chk(bull_t)} Bull Trend (EMA+ADX)")
        cc2.markdown(f"{chk(ltouch)} SD Band Touch ▲")
        cc3.markdown(f"{chk(c_p > v)} Close > VWAP")
        cc4.markdown(f"{chk(wbull)} Weekly VWAP ▲")
        cc1.markdown(f"{chk(bear_t)} Bear Trend (EMA+ADX)")
        cc2.markdown(f"{chk(stouch)} SD Band Touch ▼")
        cc3.markdown(f"{chk(c_p < v)} Close < VWAP")
        cc4.markdown(f"{chk(wbear)} Weekly VWAP ▼")
        cc1.markdown(f"{chk(dlt >= delta_min_str)} Delta Long ≥{delta_min_str}")
        cc2.markdown(f"{chk(dlt <= -delta_min_str)} Delta Short")
        cc3.markdown(f"{chk(vplong)} Above VAL  {chk(vpshrt)} Below VAH")
        cc4.markdown(f"ADX {adx:.1f}  Delta {dlt:+.3f}")

    st.markdown("---")

    # ── Main chart ───────────────────────────────────────────────────────────
    CHART_BARS = 150
    df_c    = df.tail(CHART_BARS).reset_index(drop=True)
    vwap_c  = vwap_df.tail(CHART_BARS).reset_index(drop=True)
    ema_c   = ema_ser.tail(CHART_BARS).reset_index(drop=True)
    w_c     = w_vwap.tail(CHART_BARS).reset_index(drop=True)
    sig_c   = signals.tail(CHART_BARS).reset_index(drop=True)
    delta_c = delta_s.tail(CHART_BARS).reset_index(drop=True)
    adx_c   = adx_ser.tail(CHART_BARS).reset_index(drop=True)

    fig = make_subplots(
        rows=3, cols=1, row_heights=[0.60, 0.20, 0.20],
        shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=("Price + VWAP Bands", "Delta Ratio", "ADX"),
    )
    fig.add_trace(go.Candlestick(
        x=df_c["open_time"], open=df_c["open"], high=df_c["high"],
        low=df_c["low"], close=df_c["close"],
        increasing_line_color="#56d364", decreasing_line_color="#ff7b72",
        name="Price", showlegend=False,
    ), row=1, col=1)
    for col_name, colour, dash, opacity, label in [
        ("upper2","#ff7b72","dash", 0.4, f"+2σ (×{sd2_multi})"),
        ("upper1","#ffa657","dot",  0.6, f"+1σ (×{sd1_multi})"),
        ("vwap",  "#58a6ff","solid",1.0, "VWAP"),
        ("lower1","#ffa657","dot",  0.6, f"-1σ (×{sd1_multi})"),
        ("lower2","#ff7b72","dash", 0.4, f"-2σ (×{sd2_multi})"),
    ]:
        fig.add_trace(go.Scatter(
            x=df_c["open_time"], y=vwap_c[col_name], mode="lines",
            line=dict(color=colour, dash=dash,
                      width=1.5 if col_name == "vwap" else 1.0),
            opacity=opacity, name=label,
        ), row=1, col=1)
    if use_weekly:
        fig.add_trace(go.Scatter(
            x=df_c["open_time"], y=w_c, mode="lines",
            line=dict(color="#bc8cff", dash="longdash", width=1.2),
            opacity=0.7, name="Weekly VWAP",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_c["open_time"], y=ema_c, mode="lines",
        line=dict(color="#d29922", width=1.2, dash="dot"),
        opacity=0.8, name=f"EMA {ema_period}",
    ), row=1, col=1)
    if use_vol_profile and poc > 0:
        for price, colour, label in [
            (poc, "#e3b341", "POC"),
            (vah, "#79c0ff", "VAH"),
            (val, "#79c0ff", "VAL"),
        ]:
            fig.add_hline(y=price,
                          line=dict(color=colour, dash="dot", width=1.0),
                          annotation_text=f" {label} ${price:,.0f}",
                          annotation_font_color=colour, row=1, col=1)
    long_mask  = sig_c == "LONG"
    short_mask = sig_c == "SHORT"
    if long_mask.any():
        fig.add_trace(go.Scatter(
            x=df_c["open_time"][long_mask],
            y=df_c["low"][long_mask] * 0.9985,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#56d364"),
            name="Long Signal",
        ), row=1, col=1)
    if short_mask.any():
        fig.add_trace(go.Scatter(
            x=df_c["open_time"][short_mask],
            y=df_c["high"][short_mask] * 1.0015,
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ff7b72"),
            name="Short Signal",
        ), row=1, col=1)
    if open_trade is not None:
        ot = open_trade
        for price, colour, label in [
            (ot["sl"],    "#ff7b72", f"SL ${ot['sl']:,.0f}"),
            (ot["tp"],    "#56d364", f"TP ${ot['tp']:,.0f}"),
            (ot["entry"], "#58a6ff", f"Entry ${ot['entry']:,.0f}"),
        ]:
            fig.add_hline(y=price,
                          line=dict(color=colour, dash="dash", width=1.5),
                          annotation_text=f" {label}",
                          annotation_font_color=colour, row=1, col=1)
        if use_partial and ot.get("first_target"):
            fig.add_hline(y=ot["first_target"],
                          line=dict(color="#e3b341", dash="dashdot", width=1.0),
                          annotation_text=f" Partial {partial_pct}%",
                          annotation_font_color="#e3b341", row=1, col=1)
    colours_d = ["#56d364" if v >= 0 else "#ff7b72" for v in delta_c]
    fig.add_trace(go.Bar(x=df_c["open_time"], y=delta_c, marker_color=colours_d,
                         name="Delta", showlegend=False), row=2, col=1)
    fig.add_hline(y=delta_min_str,  line=dict(color="#56d364", dash="dot", width=0.8), row=2, col=1)
    fig.add_hline(y=-delta_min_str, line=dict(color="#ff7b72", dash="dot", width=0.8), row=2, col=1)
    fig.add_hline(y=0,              line=dict(color="#8b949e", width=0.5),             row=2, col=1)
    fig.add_trace(go.Scatter(x=df_c["open_time"], y=adx_c, mode="lines",
                             line=dict(color="#bc8cff", width=1.3),
                             name="ADX", showlegend=False), row=3, col=1)
    fig.add_hline(y=adx_min, line=dict(color="#d29922", dash="dash", width=0.8), row=3, col=1)
    fig.update_layout(
        height=680, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                    orientation="h", y=1.03, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    for axis in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
        fig.update_layout(**{axis: dict(gridcolor="#21262d", showgrid=True,
                                        zeroline=False, linecolor="#30363d")})
    st.plotly_chart(fig, use_container_width=True)

    # ── P&L section ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📊 PAPER TRADING P&L</p>',
                unsafe_allow_html=True)

    eq_data   = st.session_state["pt_equity_curve"]
    eq_times  = [p["time"] if hasattr(p["time"], "strftime")
                 else datetime.fromisoformat(str(p["time"]))
                 for p in eq_data]
    eq_values = [p["equity"] for p in eq_data]
    if open_trade is not None:
        eq_times.append(datetime.now(timezone.utc))
        eq_values.append(equity)

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=eq_times, y=eq_values, mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        fill="tonexty", fillcolor="rgba(88,166,255,0.10)",
        marker=dict(size=6, color=[
            "#56d364" if v >= START_EQUITY else "#ff7b72" for v in eq_values
        ]),
        name="Equity",
    ))
    eq_fig.add_hline(y=START_EQUITY,
                     line=dict(color="#8b949e", dash="dash", width=1),
                     annotation_text=f" Start ${START_EQUITY:,.0f}",
                     annotation_font_color="#8b949e")
    _y_min = min(eq_values) if eq_values else START_EQUITY
    _y_max = max(eq_values) if eq_values else START_EQUITY
    _y_pad = max((_y_max - _y_min) * 0.3, START_EQUITY * 0.005)
    eq_fig.update_layout(
        title="Equity Curve", height=280,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d",
                   range=[_y_min - _y_pad, _y_max + _y_pad]),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    if not closed_trades:
        st.caption("No closed trades yet — waiting for first signal.")

    # ── Trade log ────────────────────────────────────────────────────────────
    st.markdown("**Trade Log** — all columns · newest first · saved to disk")
    if not closed_trades and open_trade is None:
        st.caption("No trades recorded this session.")
    else:
        rows = []
        if open_trade is not None:
            ot    = open_trade
            pnl_r = ((live_price - ot["entry"]) / ot["sl_dist"]
                     if ot["direction"] == "LONG"
                     else (ot["entry"] - live_price) / ot["sl_dist"])
            ot_notional = ot["entry"] * ot["lots"]
            rows.append({
                "Time":     ot["entry_time"].strftime("%m-%d %H:%M"),
                "Dir":      ot["direction"],
                "Entry":    f"${ot['entry']:,.2f}",
                "Notional": f"${ot_notional:,.2f}",
                "Risk $":   f"${ot['risk_usd']:,.2f}",
                "SL":       f"${ot['sl']:,.2f}",
                "TP":       f"${ot['tp']:,.2f}",
                "Exit":     "—",
                "Result":   "OPEN",
                "P&L $":    f"${unreal:+,.2f}",
                "P&L R":    f"{pnl_r:+.2f}R",
                "Balance":  "—",
            })
        for t in reversed(closed_trades):
            et = t["entry_time"]
            ts = et.strftime("%m-%d %H:%M") if hasattr(et, "strftime") else str(et)[:16]
            rows.append({
                "Time":     ts,
                "Dir":      t["direction"],
                "Entry":    f"${t['entry']:,.2f}",
                "Notional": f"${t.get('notional', t['entry'] * t.get('lots', 0)):,.2f}",
                "Risk $":   f"${t.get('risk_usd', 0):,.2f}",
                "SL":       f"${t['sl']:,.2f}",
                "TP":       f"${t['tp']:,.2f}",
                "Exit":     f"${t['exit_price']:,.2f}",
                "Result":   t["result"],
                "P&L $":    f"${t['pnl_usd']:+,.2f}",
                "P&L R":    f"{t['pnl_r']:+.2f}R",
                "Balance":  f"${t['balance']:,.2f}",
            })

        log_df = pd.DataFrame(rows)

        def colour_log(row):
            if row["Result"] == "WIN":  return ["background-color:#1a472a"] * len(row)
            if row["Result"] == "LOSS": return ["background-color:#4a1a1a"] * len(row)
            if row["Result"] == "OPEN": return ["background-color:#1a2a47"] * len(row)
            return [""] * len(row)

        st.dataframe(log_df.style.apply(colour_log, axis=1),
                     use_container_width=True, hide_index=True, height=400)

    # ── Alert feed ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🔔 SIGNAL ALERT FEED</p>',
                unsafe_allow_html=True)
    alerts = st.session_state["alerts"]
    if not alerts:
        st.caption("No signals detected this session.")
    else:
        num_cols = min(len(alerts), 4)
        cols_a   = st.columns(num_cols)
        for idx, a in enumerate(list(alerts)[:4]):
            cls   = "alert-long" if a["direction"] == "LONG" else "alert-short"
            arrow = "▲" if a["direction"] == "LONG" else "▼"
            cols_a[idx % num_cols].markdown(
                f'<div class="alert-row {cls}">'
                f'{a["time"]} &nbsp; {arrow} <b>{a["direction"]}</b><br>'
                f'@ ${a["price"]:,.2f} &nbsp;|&nbsp; ATR ${a["atr"]:,.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ot_now = st.session_state.get("pt_open_trade")
    if ot_now is not None:
        pnl_r_now = ((live_price - ot_now["entry"]) / ot_now["sl_dist"]
                     if ot_now["direction"] == "LONG"
                     else (ot_now["entry"] - live_price) / ot_now["sl_dist"])
        st.markdown(
            f'<p style="color:#58a6ff; font-size:0.85rem;">'
            f'⚡ Live — updates every {TICK_INTERVAL}s automatically · '
            f'{now_utc} · Capstone — Jun Hyoung Park (541039718)</p>',
            unsafe_allow_html=True)
    else:
        st.caption(f"⚡ Live — updates every {TICK_INTERVAL}s · {now_utc} · "
                   f"Capstone — Jun Hyoung Park (541039718)")

    # ── Recent trade closed banner ────────────────────────────────────────────
    if closed_trades:
        last_c = closed_trades[-1]
        exit_t = last_c.get("exit_time")
        if exit_t is not None and hasattr(exit_t, "timestamp"):
            age = (datetime.now(timezone.utc) - exit_t).total_seconds()
            if age < 30:
                rc = "#56d364" if last_c["result"] == "WIN" else "#ff7b72"
                st.markdown(
                    f"<span style='color:{rc}; font-weight:700;'>🔔 Trade closed: "
                    f"{last_c['result']}  ${last_c['pnl_usd']:+,.2f} "
                    f"({last_c['pnl_r']:+.2f}R)  Balance: ${last_c['balance']:,.2f}"
                    f"</span>", unsafe_allow_html=True)


# ===========================================================================
# ── Invoke the fragment
# ===========================================================================
live_dashboard()
