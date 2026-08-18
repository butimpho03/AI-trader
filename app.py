"""
TradeLens AI — Phase 1 (single-file edition)

An educational, mobile-first trading chart analysis and paper-trading
platform. This does NOT give financial advice, does NOT execute real
trades, and NEVER guarantees any outcome.

Phase 0 built the foundation. Phase 1 (this version) polishes the mobile
UI: consistent empty states, status badges, and readable styling that
doesn't depend on the visitor's default Streamlit theme. Chart upload,
AI vision, and the strategy engine are still not implemented — every
placeholder below says so honestly instead of pretending to work.

WHY ONE FILE: this project is deployed and edited entirely from a phone
via the GitHub website. Multi-folder projects require git or a desktop
browser to upload correctly — a phone browser can't reliably preserve
folder structure. Keeping everything in one file removes that problem
completely. We can split this into multiple files later once editing
happens from a computer or via git.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from pydantic import BaseModel

# =====================================================================
# SECTION 1: DATA MODELS
# (equivalent to what would be tradelens/types/*.py in a multi-file build)
# =====================================================================


class AssetClass(str, Enum):
    FOREX = "forex"
    INDICES = "indices"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    STOCKS = "stocks"
    ETF = "etf"
    FUTURES = "futures"


class Instrument(BaseModel):
    """Generic instrument model. IMPORTANT: nothing in the engine/provider
    sections below should ever branch on a specific symbol like "BTC" or
    "XAUUSD" — those only appear in EXAMPLE_INSTRUMENTS as example data."""

    instrument_id: str
    display_name: str
    symbol: str
    asset_class: AssetClass
    base_asset: str
    quote_asset: Optional[str] = None
    currency: Optional[str] = None
    data_provider: str = "seed"


class TimeframeRole(str, Enum):
    HTF = "HTF"
    MTF = "MTF"
    LTF = "LTF"


class TradingMode(str, Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EvidenceLayer(str, Enum):
    """Each strategy belongs to one layer. This is what stops strategies
    from just "fighting" each other later — each layer answers a
    different question, and the (future) ConfluenceEngine combines them
    in order rather than averaging arbitrary votes."""

    MARKET_CONTEXT = "market_context"
    MARKET_LOCATION = "market_location"
    LIQUIDITY_STRUCTURE = "liquidity_structure"
    CONFIRMATION = "confirmation"
    EXECUTION = "execution"


class PriceZone(BaseModel):
    label: str
    low: float
    high: float


class StrategyResult(BaseModel):
    """The structured output every future strategy module must produce.
    Not implemented in Phase 0 — this shape exists so later phases have
    a real contract to build against."""

    strategy_id: str
    strategy_name: str
    layer: EvidenceLayer
    timeframe_role: TimeframeRole
    bias: Bias
    evidence: List[str] = []
    invalidation: Optional[str] = None
    zones: List[PriceZone] = []
    quality_contribution: float = 0.0  # NOT a win probability
    insufficient_data: bool = False
    notes: List[str] = []


class SetupQualityBreakdown(BaseModel):
    htf_alignment: float = 0.0
    mtf_alignment: float = 0.0
    ltf_confirmation: float = 0.0
    liquidity: float = 0.0
    structure: float = 0.0
    location: float = 0.0
    risk_reward: float = 0.0
    total: float = 0.0


class ChartQuality(str, Enum):
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class ChartUpload(BaseModel):
    chart_id: str
    session_id: str
    instrument_id: str
    timeframe: Optional[str] = None
    timeframe_role: TimeframeRole
    image_ref: str
    quality: ChartQuality = ChartQuality.UNKNOWN


class AnalysisStatus(str, Enum):
    WAITING_FOR_CHARTS = "WAITING_FOR_CHARTS"
    READY = "READY"
    ANALYZING = "ANALYZING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class AnalysisSession(BaseModel):
    """One session = exactly one instrument + up to three charts
    (HTF/MTF/LTF). Sessions are the boundary that prevents charts from
    different instruments ever being combined — enforced later by the
    session store, not just convention."""

    session_id: str
    instrument_id: str
    htf_chart_id: Optional[str] = None
    mtf_chart_id: Optional[str] = None
    ltf_chart_id: Optional[str] = None
    status: AnalysisStatus = AnalysisStatus.WAITING_FOR_CHARTS


# =====================================================================
# SECTION 2: AI PROVIDER ABSTRACTION
# (equivalent to tradelens/types/ai_provider.py + providers/ai_provider.py)
# =====================================================================


class AIProviderCapabilities(BaseModel):
    supports_vision: bool
    supports_text: bool
    provider_name: str
    is_free_tier: bool = True


class ChartInterpretationResult(BaseModel):
    success: bool
    failure_reason: Optional[str] = None


class ExplanationResult(BaseModel):
    success: bool
    text: Optional[str] = None
    failure_reason: Optional[str] = None


class AIProvider(ABC):
    @abstractmethod
    def get_capabilities(self) -> AIProviderCapabilities: ...

    @abstractmethod
    def interpret_chart(self, image_ref: str) -> ChartInterpretationResult: ...

    @abstractmethod
    def explain(self, context: Dict[str, Any], instruction: str) -> ExplanationResult: ...


class NoOpAIProvider(AIProvider):
    """Default provider. Honestly reports that no AI is connected rather
    than pretending to analyze a chart.

    FREE-TIER PLAN FOR A LATER PHASE (not implemented yet): both Google
    AI Studio (Gemini API) and Groq currently offer free-tier
    vision-capable models with no card required. Whichever we pick later
    becomes one more class implementing AIProvider — nothing else in
    this file changes.
    """

    def get_capabilities(self) -> AIProviderCapabilities:
        return AIProviderCapabilities(
            supports_vision=False,
            supports_text=False,
            provider_name="None configured",
        )

    def interpret_chart(self, image_ref: str) -> ChartInterpretationResult:
        return ChartInterpretationResult(
            success=False,
            failure_reason="No AI provider is configured yet.",
        )

    def explain(self, context: Dict[str, Any], instruction: str) -> ExplanationResult:
        return ExplanationResult(
            success=False,
            failure_reason="No AI provider is configured yet.",
        )


def get_configured_ai_provider() -> AIProvider:
    selected = os.environ.get("AI_PROVIDER", "none")
    if selected == "none":
        return NoOpAIProvider()
    return NoOpAIProvider()  # unimplemented provider requested -> fail honestly


class MarketDataProvider(ABC):
    provider_name: str
    is_connected: bool

    @abstractmethod
    def get_price(self, instrument_id: str) -> Optional[float]: ...


class NoOpMarketDataProvider(MarketDataProvider):
    def __init__(self) -> None:
        self.provider_name = "None configured"
        self.is_connected = False

    def get_price(self, instrument_id: str) -> Optional[float]:
        return None


def get_configured_market_data_provider() -> MarketDataProvider:
    return NoOpMarketDataProvider()


# =====================================================================
# SECTION 3: STRATEGY REGISTRY
# (equivalent to tradelens/engines/strategy_registry.py)
# Ships EMPTY on purpose — implementing strategies before the AI vision
# layer exists would mean fabricating analysis logic.
# =====================================================================

REGISTERED_STRATEGIES: List[str] = []


# =====================================================================
# SECTION 4: INSTRUMENT ARCHITECTURE (Phase 2)
# (equivalent to tradelens/instruments/* + an InstrumentResolver)
#
# EXAMPLE_INSTRUMENTS below is DATA, not logic — nothing above this
# section (engines, providers) is allowed to read from it, and no
# strategy/engine code is allowed to branch on any symbol here. This is
# still a fixed example list, NOT a live feed — no MarketDataProvider is
# connected yet, so this cannot be a complete market list. What IS real
# in this phase: the resolver logic (aliases, normalization, manual
# override) that a live provider will plug into later without this
# section needing to change shape.
# =====================================================================


class InstrumentDraft(BaseModel):
    """What we have before an instrument is confidently resolved — e.g.
    right after the user types a symbol we don't recognize. Manual
    confirmation of a draft always overrides any AI guess (there is no
    AI guessing yet, since chart vision isn't built, but this is the
    same override rule the project brief requires for that later)."""

    symbol_input: str
    asset_class_hint: Optional[AssetClass] = None


def _mk(instrument_id, display_name, symbol, asset_class, base_asset, quote_asset, aliases=None) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        display_name=display_name,
        symbol=symbol,
        asset_class=asset_class,
        base_asset=base_asset,
        quote_asset=quote_asset,
        currency=quote_asset,
    )


# Alternate tickers different brokers/providers use for the same
# instrument (per the project brief's "provider symbol resolution" —
# e.g. Nasdaq 100 may appear as NAS100, USTEC, NQ100, NASDAQ100). This
# is what INSTRUMENT_ALIASES below encodes, keyed by our canonical
# instrument_id.
INSTRUMENT_ALIASES: Dict[str, List[str]] = {
    "nas100": ["USTEC", "NQ100", "NASDAQ100", "NDX"],
    "us30": ["DJ30", "DOW", "DJIA"],
    "spx500": ["US500", "SPX", "SP500"],
    "ger40": ["DAX", "DE40", "DAX40"],
    "uk100": ["FTSE", "FTSE100"],
    "jp225": ["NIKKEI", "NKY"],
    "xauusd": ["GOLD"],
    "xagusd": ["SILVER"],
    "btcusdt": ["BTCUSD", "XBTUSD"],
    "ethusdt": ["ETHUSD"],
}

EXAMPLE_INSTRUMENTS: List[Instrument] = [
    # --- Forex majors & a few minors/exotics ---
    _mk("eurusd", "Euro / US Dollar", "EUR/USD", AssetClass.FOREX, "EUR", "USD"),
    _mk("gbpusd", "British Pound / US Dollar", "GBP/USD", AssetClass.FOREX, "GBP", "USD"),
    _mk("usdjpy", "US Dollar / Japanese Yen", "USD/JPY", AssetClass.FOREX, "USD", "JPY"),
    _mk("audusd", "Australian Dollar / US Dollar", "AUD/USD", AssetClass.FOREX, "AUD", "USD"),
    _mk("usdcad", "US Dollar / Canadian Dollar", "USD/CAD", AssetClass.FOREX, "USD", "CAD"),
    _mk("usdchf", "US Dollar / Swiss Franc", "USD/CHF", AssetClass.FOREX, "USD", "CHF"),
    _mk("nzdusd", "New Zealand Dollar / US Dollar", "NZD/USD", AssetClass.FOREX, "NZD", "USD"),
    _mk("eurgbp", "Euro / British Pound", "EUR/GBP", AssetClass.FOREX, "EUR", "GBP"),
    _mk("gbpjpy", "British Pound / Japanese Yen", "GBP/JPY", AssetClass.FOREX, "GBP", "JPY"),
    _mk("eurjpy", "Euro / Japanese Yen", "EUR/JPY", AssetClass.FOREX, "EUR", "JPY"),
    # --- Indices ---
    _mk("nas100", "Nasdaq 100 Index", "NAS100", AssetClass.INDICES, "NAS100", None),
    _mk("us30", "Dow Jones 30 Index", "US30", AssetClass.INDICES, "US30", None),
    _mk("spx500", "S&P 500 Index", "SPX500", AssetClass.INDICES, "SPX500", None),
    _mk("ger40", "Germany 40 Index", "GER40", AssetClass.INDICES, "GER40", None),
    _mk("uk100", "UK 100 Index", "UK100", AssetClass.INDICES, "UK100", None),
    _mk("jp225", "Japan 225 Index", "JP225", AssetClass.INDICES, "JP225", None),
    # --- Crypto ---
    _mk("btcusdt", "Bitcoin / Tether", "BTC/USDT", AssetClass.CRYPTO, "BTC", "USDT"),
    _mk("ethusdt", "Ethereum / Tether", "ETH/USDT", AssetClass.CRYPTO, "ETH", "USDT"),
    _mk("solusdt", "Solana / Tether", "SOL/USDT", AssetClass.CRYPTO, "SOL", "USDT"),
    _mk("xrpusdt", "XRP / Tether", "XRP/USDT", AssetClass.CRYPTO, "XRP", "USDT"),
    # --- Commodities ---
    _mk("xauusd", "Gold / US Dollar", "XAU/USD", AssetClass.COMMODITIES, "XAU", "USD"),
    _mk("xagusd", "Silver / US Dollar", "XAG/USD", AssetClass.COMMODITIES, "XAG", "USD"),
    _mk("wti", "WTI Crude Oil", "WTI", AssetClass.COMMODITIES, "WTI", None),
    _mk("natgas", "Natural Gas", "NATGAS", AssetClass.COMMODITIES, "NATGAS", None),
    # --- Stocks ---
    _mk("aapl", "Apple Inc.", "AAPL", AssetClass.STOCKS, "AAPL", None),
    _mk("msft", "Microsoft Corp.", "MSFT", AssetClass.STOCKS, "MSFT", None),
    _mk("nvda", "NVIDIA Corp.", "NVDA", AssetClass.STOCKS, "NVDA", None),
    _mk("tsla", "Tesla Inc.", "TSLA", AssetClass.STOCKS, "TSLA", None),
    _mk("amzn", "Amazon.com Inc.", "AMZN", AssetClass.STOCKS, "AMZN", None),
]


def _normalize(text: str) -> str:
    """Strip spaces/slashes/punctuation and lowercase, so 'EUR/USD',
    'eur usd', and 'eurusd' all normalize the same way for matching."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


class InstrumentResolver:
    """USER SYMBOL -> CANONICAL INSTRUMENT -> (future) PROVIDER SYMBOL.

    Deterministic, code-driven lookup — never a guess. If nothing
    matches, callers must fall back to a manual InstrumentDraft rather
    than assuming; see page_analyze()'s "add manually" flow.
    """

    def __init__(self, instruments: List[Instrument], aliases: Dict[str, List[str]]):
        self._instruments = instruments
        self._aliases = aliases
        self._by_id = {i.instrument_id: i for i in instruments}

    def resolve_exact(self, user_input: str) -> Optional[Instrument]:
        norm = _normalize(user_input)
        if not norm:
            return None
        for inst in self._instruments:
            if _normalize(inst.symbol) == norm or _normalize(inst.instrument_id) == norm:
                return inst
        for inst_id, alias_list in self._aliases.items():
            if any(_normalize(a) == norm for a in alias_list):
                return self._by_id.get(inst_id)
        return None

    def search(self, query: str) -> List[Instrument]:
        q = query.strip().lower()
        if not q:
            return self._instruments
        norm_q = _normalize(query)
        matches: List[Instrument] = []
        for inst in self._instruments:
            hit = (
                q in inst.symbol.lower()
                or q in inst.display_name.lower()
                or norm_q in _normalize(inst.symbol)
            )
            if not hit:
                for alias in self._aliases.get(inst.instrument_id, []):
                    if norm_q in _normalize(alias):
                        hit = True
                        break
            if hit:
                matches.append(inst)
        return matches

    def aliases_for(self, instrument_id: str) -> List[str]:
        return self._aliases.get(instrument_id, [])


instrument_resolver = InstrumentResolver(EXAMPLE_INSTRUMENTS, INSTRUMENT_ALIASES)


# =====================================================================
# SECTION 5: UI STYLING HELPERS
# =====================================================================

MOBILE_CSS = """
<style>
/* Force a dark app background regardless of the visitor's default
   Streamlit theme, since this app ships without a .streamlit/config.toml. */
[data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background-color: #0b0f14 !important;
}
.block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 480px; }
.block-container, .block-container p, .block-container span, .block-container label,
h1, h2, h3, h4 { color: #e6ebf1 !important; }

/* Cards */
.tl-card { background-color: #121821; border: 1px solid #26313f; border-radius: 14px;
    padding: 16px; margin-bottom: 12px; }
.tl-card-row { display: flex; align-items: flex-start; gap: 12px; }
.tl-card-icon { font-size: 1.3rem; line-height: 1.3; }
.tl-card h3 { margin-top: 0; margin-bottom: 6px; font-size: 1rem; color: #e6ebf1 !important; }
.tl-card p { margin: 0; font-size: 0.88rem; color: #8b97a6 !important; line-height: 1.45; }

/* Empty-state cards (dashed border, centered, used on placeholder pages) */
.tl-empty { border: 1.5px dashed #26313f; border-radius: 14px; padding: 28px 16px;
    text-align: center; margin-bottom: 12px; }
.tl-empty .tl-empty-icon { font-size: 1.8rem; margin-bottom: 8px; }
.tl-empty h3 { margin: 0 0 6px 0; font-size: 0.95rem; color: #e6ebf1 !important; }
.tl-empty p { margin: 0; font-size: 0.85rem; color: #8b97a6 !important; line-height: 1.45; }

/* Badges / status pills */
.tl-badge { display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; border: 1px solid; white-space: nowrap; }
.tl-badge-neutral { color: #8b97a6; border-color: #26313f; background: #0b0f14; }
.tl-badge-bullish { color: #2dbd8e; border-color: #2dbd8e55; background: #2dbd8e18; }
.tl-badge-accent  { color: #4f8cff; border-color: #4f8cff55; background: #4f8cff18; }

.tl-status-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }

.tl-app-title { font-size: 0.8rem; letter-spacing: 0.08em; color: #8b97a6 !important;
    font-weight: 600; text-transform: uppercase; margin-bottom: 2px; }
.tl-page-title { font-size: 1.4rem; font-weight: 700; margin: 0 0 14px 0; color: #e6ebf1 !important; }

.tl-footer { text-align: center; font-size: 0.72rem; color: #4a5568 !important;
    margin-top: 28px; padding-top: 14px; border-top: 1px solid #1b2330; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #121821 !important; border-right: 1px solid #26313f; }
section[data-testid="stSidebar"] * { color: #e6ebf1 !important; }
section[data-testid="stSidebar"] label { font-size: 0.92rem; padding: 4px 0; }
</style>
"""


def page_header(title: str) -> None:
    st.markdown(f'<div class="tl-page-title">{title}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str, icon: str = "") -> None:
    icon_html = f'<span class="tl-card-icon">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="tl-card"><div class="tl-card-row">{icon_html}'
        f'<div><h3>{title}</h3><p>{body_html}</p></div></div></div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="tl-empty"><div class="tl-empty-icon">{icon}</div>'
        f'<h3>{title}</h3><p>{body_html}</p></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="tl-badge tl-badge-{tone}">{text}</span>'


def footer() -> None:
    st.markdown(
        '<div class="tl-footer">TradeLens AI · Phase 1 · Educational tool, not financial advice</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# SECTION 6: PAGES
# Each function renders one "page". Navigation switches between them
# via a sidebar radio button — no pages/ folder needed.
# =====================================================================


def page_dashboard() -> None:
    page_header("Dashboard")

    ai_caps = get_configured_ai_provider().get_capabilities()
    market_data = get_configured_market_data_provider()
    st.markdown(
        '<div class="tl-status-row">'
        + badge(f"AI: {ai_caps.provider_name}", "bullish" if ai_caps.supports_vision else "neutral")
        + badge(f"Data: {market_data.provider_name}", "bullish" if market_data.is_connected else "neutral")
        + badge(f"{len(REGISTERED_STRATEGIES)} strategies", "neutral")
        + badge(f"{len(EXAMPLE_INSTRUMENTS)} example instruments", "accent")
        + "</div>",
        unsafe_allow_html=True,
    )

    empty_state(
        "📭",
        "No active analyses yet",
        "Sessions you start from <b>Analyze</b> will appear here, each tracking "
        "its own HTF / MTF / LTF charts independently.",
    )
    card(
        "Phase 1 status",
        "Mobile UI and navigation are in place. Chart upload, AI vision, and the "
        "strategy engine are not implemented yet — see Settings for what's connected.",
        icon="🧭",
    )
    card(
        "Important",
        "TradeLens AI is an educational analysis and paper-trading tool. "
        "It does not execute real trades and does not guarantee any outcome.",
        icon="⚠️",
    )
    footer()


def page_analyze() -> None:
    page_header("Analyze")
    st.markdown("**Instrument**")
    query = st.text_input(
        "Search symbol, name or market",
        placeholder="e.g. EUR/USD, NAS100, USTEC, BTC/USDT",
        label_visibility="collapsed",
    )

    if query:
        exact = instrument_resolver.resolve_exact(query)
        if exact:
            aliases = instrument_resolver.aliases_for(exact.instrument_id)
            alias_note = f" · also known as: {', '.join(aliases)}" if aliases else ""
            card(
                f"{exact.symbol} resolved",
                f"{exact.display_name} · {exact.asset_class.value}{alias_note}",
                icon="✅",
            )
        else:
            results = instrument_resolver.search(query)
            if results:
                st.caption(f"{len(results)} possible match(es):")
                for inst in results[:8]:
                    st.write(f"**{inst.symbol}** — {inst.display_name}")
            else:
                empty_state(
                    "❓",
                    "Not in the example list",
                    f"\"{query}\" isn't recognized yet. This is only a small example "
                    "list, not a live market feed — real resolution across every "
                    "supported market arrives once a data provider is connected.",
                )
                with st.expander("Add it manually instead"):
                    st.caption(
                        "Your manual entry always overrides automatic matching — "
                        "same rule that will apply later to AI chart-reading guesses."
                    )
                    manual_class = st.selectbox(
                        "Asset class", [c.value for c in AssetClass], key="manual_asset_class"
                    )
                    if st.button("Use this instrument for now"):
                        st.session_state["manual_instrument"] = InstrumentDraft(
                            symbol_input=query, asset_class_hint=AssetClass(manual_class)
                        )
                        st.success(f"Using \"{query}\" ({manual_class}) — unresolved, but noted.")

    st.caption(
        "Searching a small example list across forex, indices, crypto, "
        "commodities, and stocks — not a live provider yet."
    )
    st.markdown("&nbsp;", unsafe_allow_html=True)
    card("Mode", "Scalping / Day Trading / Swing / Analyze All — coming in a later phase.", icon="⏱️")
    card("Charts", "HTF / MTF / LTF upload and quality validation — coming in a later phase.", icon="🖼️")
    empty_state("🚧", "Not runnable yet", "This page is a placeholder. No analysis runs yet.")
    footer()


def page_active_setups() -> None:
    page_header("Active Setups")
    empty_state(
        "📭",
        "No sessions yet",
        "Once sessions exist, each one's status (WAITING FOR CHARTS / READY / "
        "ANALYZING / COMPLETE) will be listed here.",
    )
    footer()


def page_strategy_lab() -> None:
    page_header("Strategy Lab")
    st.markdown(
        '<div class="tl-status-row">' + badge(f"{len(REGISTERED_STRATEGIES)} registered", "neutral") + "</div>",
        unsafe_allow_html=True,
    )
    empty_state(
        "🧪",
        "Registry is empty — by design",
        "No strategies are implemented in Phase 0/1. Modules (trend following, "
        "BOS/CHoCH, FVG, RSI divergence, etc.) get added one at a time in later phases.",
    )
    card(
        "Backtesting",
        "Comes later, once real strategy modules and historical data exist. "
        "No fabricated results will ever be shown here.",
        icon="📊",
    )
    footer()


def page_paper_trading() -> None:
    page_header("Paper Trading")
    empty_state(
        "📝",
        "Coming later",
        "Paper-trading simulation (no real money, no broker execution) arrives "
        "once hypothetical setups can actually be generated.",
    )
    footer()


def page_journal() -> None:
    page_header("Journal")
    empty_state(
        "📓",
        "Coming later",
        "The trading journal (setups, charts, outcomes, filtering) arrives "
        "once paper trading exists to log from.",
    )
    footer()


def page_settings() -> None:
    page_header("Settings")
    ai = get_configured_ai_provider()
    caps = ai.get_capabilities()
    market_data = get_configured_market_data_provider()

    card(
        "AI Provider",
        badge(caps.provider_name, "bullish" if caps.supports_vision else "neutral")
        + f"<br><br>Vision analysis: {'available' if caps.supports_vision else 'not available'}. "
        "Set AI_PROVIDER in Streamlit Cloud Secrets to connect a free-tier provider later.",
        icon="🤖",
    )
    card(
        "Market Data Provider",
        badge(market_data.provider_name, "bullish" if market_data.is_connected else "neutral")
        + "<br><br>"
        + ("Live data connected." if market_data.is_connected
           else "Not connected — the app runs in screenshot-only mode."),
        icon="📡",
    )
    card(
        "About",
        "TradeLens AI is an educational chart-analysis and paper-trading tool. "
        "It never guarantees outcomes and can always report \"NO VALID SETUP\".",
        icon="ℹ️",
    )
    footer()


# =====================================================================
# SECTION 7: APP ENTRY POINT / NAVIGATION
# =====================================================================

st.set_page_config(page_title="TradeLens AI", page_icon="📊", layout="centered")
st.markdown(MOBILE_CSS, unsafe_allow_html=True)
st.markdown('<div class="tl-app-title">📊 TRADELENS AI</div>', unsafe_allow_html=True)

PAGES = {
    "🏠 Dashboard": page_dashboard,
    "🔍 Analyze": page_analyze,
    "📋 Active Setups": page_active_setups,
    "🧪 Strategy Lab": page_strategy_lab,
    "📝 Paper Trading": page_paper_trading,
    "📓 Journal": page_journal,
    "⚙️ Settings": page_settings,
}

selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[selection]()
