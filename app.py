"""
TradeLens AI — Phase 0 (single-file edition)

An educational, mobile-first trading chart analysis and paper-trading
platform. This does NOT give financial advice, does NOT execute real
trades, and NEVER guarantees any outcome.

This is Phase 0: the foundation only. Chart upload, AI vision, and the
strategy engine are not implemented yet. Every placeholder below says so
honestly instead of pretending to work.

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
    "XAUUSD" — those only appear in SEED_INSTRUMENTS as example data."""

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
# SECTION 4: EXAMPLE INSTRUMENT DATA
# (equivalent to tradelens/instruments/seed_instruments.py)
# DATA ONLY — nothing above this section is allowed to read from it.
# =====================================================================


def _mk(instrument_id, display_name, symbol, asset_class, base_asset, quote_asset) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        display_name=display_name,
        symbol=symbol,
        asset_class=asset_class,
        base_asset=base_asset,
        quote_asset=quote_asset,
        currency=quote_asset,
    )


SEED_INSTRUMENTS: List[Instrument] = [
    _mk("eurusd", "Euro / US Dollar", "EUR/USD", AssetClass.FOREX, "EUR", "USD"),
    _mk("gbpusd", "British Pound / US Dollar", "GBP/USD", AssetClass.FOREX, "GBP", "USD"),
    _mk("usdjpy", "US Dollar / Japanese Yen", "USD/JPY", AssetClass.FOREX, "USD", "JPY"),
    _mk("gbpjpy", "British Pound / Japanese Yen", "GBP/JPY", AssetClass.FOREX, "GBP", "JPY"),
    _mk("nas100", "Nasdaq 100 Index", "NAS100", AssetClass.INDICES, "NAS100", None),
    _mk("us30", "Dow Jones 30 Index", "US30", AssetClass.INDICES, "US30", None),
    _mk("spx500", "S&P 500 Index", "SPX500", AssetClass.INDICES, "SPX500", None),
    _mk("btcusdt", "Bitcoin / Tether", "BTC/USDT", AssetClass.CRYPTO, "BTC", "USDT"),
    _mk("ethusdt", "Ethereum / Tether", "ETH/USDT", AssetClass.CRYPTO, "ETH", "USDT"),
    _mk("xauusd", "Gold / US Dollar", "XAU/USD", AssetClass.COMMODITIES, "XAU", "USD"),
    _mk("xagusd", "Silver / US Dollar", "XAG/USD", AssetClass.COMMODITIES, "XAG", "USD"),
    _mk("aapl", "Apple Inc.", "AAPL", AssetClass.STOCKS, "AAPL", None),
    _mk("nvda", "NVIDIA Corp.", "NVDA", AssetClass.STOCKS, "NVDA", None),
]


def search_seed_instruments(query: str) -> List[Instrument]:
    q = query.strip().lower()
    if not q:
        return SEED_INSTRUMENTS
    return [i for i in SEED_INSTRUMENTS if q in i.symbol.lower() or q in i.display_name.lower()]


# =====================================================================
# SECTION 5: UI STYLING HELPERS
# =====================================================================

MOBILE_CSS = """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 480px; }
.tl-card { background-color: #121821; border: 1px solid #26313f; border-radius: 14px;
    padding: 16px; margin-bottom: 12px; }
.tl-card h3 { margin-top: 0; margin-bottom: 6px; font-size: 1rem; }
.tl-card p { margin: 0; font-size: 0.88rem; color: #8b97a6; line-height: 1.4; }
.tl-badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; border: 1px solid; }
.tl-badge-neutral { color: #8b97a6; border-color: #26313f; background: #0b0f14; }
.tl-badge-bullish { color: #2dbd8e; border-color: #2dbd8e55; background: #2dbd8e18; }
.tl-app-title { font-size: 0.8rem; letter-spacing: 0.08em; color: #8b97a6;
    font-weight: 600; text-transform: uppercase; }
</style>
"""


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="tl-card"><h3>{title}</h3><p>{body_html}</p></div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="tl-badge tl-badge-{tone}">{text}</span>'


# =====================================================================
# SECTION 6: PAGES
# Each function renders one "page". Navigation switches between them
# via a sidebar radio button — no pages/ folder needed.
# =====================================================================


def page_dashboard() -> None:
    st.markdown("## Dashboard")
    card(
        "Active Analyses",
        "No analysis sessions yet. Sessions you start from <b>Analyze</b> "
        "will appear here, each tracking its own HTF / MTF / LTF charts independently.",
    )
    card(
        "Phase 0 status",
        "This is a foundation build. Chart upload, AI vision, and the "
        "strategy engine are not implemented yet — see Settings for what is connected.",
    )
    card(
        "Important",
        "TradeLens AI is an educational analysis and paper-trading tool. "
        "It does not execute real trades and does not guarantee any outcome.",
    )


def page_analyze() -> None:
    st.markdown("## Analyze")
    st.markdown("**Instrument**")
    query = st.text_input(
        "Search symbol, name or market",
        placeholder="e.g. EUR/USD, NAS100, BTC/USDT",
        label_visibility="collapsed",
    )
    if query:
        results = search_seed_instruments(query)
        if results:
            for inst in results[:8]:
                st.write(f"**{inst.symbol}** — {inst.display_name}")
        else:
            st.caption("No match in the Phase 0 example list.")
    st.caption("Only searching a small example list for now. Real instrument resolution arrives in a later phase.")
    st.markdown("---")
    card("Mode", "Scalping / Day Trading / Swing / Analyze All — coming in a later phase.")
    card("Charts", "HTF / MTF / LTF upload and quality validation — coming in a later phase.")
    st.info("This page is a placeholder. No analysis runs yet.", icon="ℹ️")


def page_active_setups() -> None:
    st.markdown("## Active Setups")
    card(
        "No sessions yet",
        "Once sessions exist, each one's status (WAITING FOR CHARTS / READY / "
        "ANALYZING / COMPLETE) will be listed here.",
    )


def page_strategy_lab() -> None:
    st.markdown("## Strategy Lab")
    card(
        "Strategy registry",
        f"<b>{len(REGISTERED_STRATEGIES)}</b> strategy modules registered.<br><br>"
        "No strategies are implemented in Phase 0 — modules (trend following, "
        "BOS/CHoCH, FVG, RSI divergence, etc.) get added one at a time in later phases.",
    )
    card(
        "Backtesting",
        "Strategy backtesting comes later, once real strategy modules and "
        "historical data exist. No fabricated results will ever be shown here.",
    )


def page_paper_trading() -> None:
    st.markdown("## Paper Trading")
    card(
        "Coming later",
        "Paper-trading simulation (no real money, no broker execution) arrives "
        "once hypothetical setups can actually be generated.",
    )


def page_journal() -> None:
    st.markdown("## Journal")
    card(
        "Coming later",
        "The trading journal (setups, charts, outcomes, filtering) arrives "
        "once paper trading exists to log from.",
    )


def page_settings() -> None:
    st.markdown("## Settings")
    ai = get_configured_ai_provider()
    caps = ai.get_capabilities()
    market_data = get_configured_market_data_provider()

    st.markdown("**AI Provider**")
    tone = "bullish" if caps.supports_vision else "neutral"
    st.markdown(badge(caps.provider_name, tone), unsafe_allow_html=True)
    st.caption(
        f"Vision analysis: {'available' if caps.supports_vision else 'not available'}. "
        "Set AI_PROVIDER in Streamlit Cloud Secrets to connect a free-tier provider later."
    )
    st.markdown("---")
    st.markdown("**Market Data Provider**")
    tone = "bullish" if market_data.is_connected else "neutral"
    st.markdown(badge(market_data.provider_name, tone), unsafe_allow_html=True)
    st.caption(
        "Live data connected." if market_data.is_connected
        else "Not connected — the app runs in screenshot-only mode."
    )
    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "TradeLens AI is an educational chart-analysis and paper-trading tool. "
        "It never guarantees outcomes and can always report \"NO VALID SETUP\"."
    )


# =====================================================================
# SECTION 7: APP ENTRY POINT / NAVIGATION
# =====================================================================

st.set_page_config(page_title="TradeLens AI", page_icon="📊", layout="centered")
st.markdown(MOBILE_CSS, unsafe_allow_html=True)
st.markdown('<div class="tl-app-title">TRADELENS AI</div>', unsafe_allow_html=True)

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
