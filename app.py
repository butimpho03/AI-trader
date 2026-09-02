"""
TradeLens AI — Quick Scan update (single-file edition)

An educational, mobile-first trading chart analysis and paper-trading
platform. This does NOT give financial advice, does NOT execute real
trades, and NEVER guarantees any outcome.

Phase 0 built the foundation. Phase 1 polished the mobile UI. Phase 2
added real instrument resolution. Phase 3 added real session
management. Phase 4 added real three-chart upload. Phase 5 added
deterministic chart quality checking. Phase 6 added real AI vision
(Gemini free tier). Phase 7 added multi-timeframe comparison. Phase 8
added a real strategy engine (one honest module). Phase 9 added the
confluence engine (a real setup-quality score, capped at 60/100 until
more evidence exists). Phase 10 added the deterministic level
calculation engine (AI reads only raw price numbers off the LTF chart;
100% Python code does all entry/stop/target math) for the Multi-
Timeframe path.

THIS UPDATE adds a second, DEFAULT path: Quick Scan. One chart, one
integrated AI call, structured JSON output (instrument, structure,
support/resistance, patterns, bias, entry/SL/TP1-3, evidence score,
invalidation, reasoning). Unlike the Multi-Timeframe path, Quick Scan
lets the AI propose entry/SL/TP directly — there's no second/third
timeframe to derive levels from with just one chart. The honesty
backstop is entirely downstream and code-only
(validate_and_normalize_quick_scan): risk/reward is always recomputed
from the AI's own numbers, never trusted as a stated ratio, and any
setup where the stop/TP land on the wrong side of entry is discarded
and shown as NO TRADE with the reason — verified against 9 test cases,
including AI-hallucination scenarios, before shipping. An unreadable
image (IMAGE_UNREADABLE) is explicitly distinguished from a readable
chart with no clean setup (NO TRADE) — these no longer collapse into a
single "unclear." The Multi-Timeframe path (Phases 6-10) is unchanged
and still available as an explicit second mode. Named technical
strategies, backtesting, paper trading, and the journal are still not
implemented — every placeholder below says so honestly instead of
pretending to work.


WHY ONE FILE: this project is deployed and edited entirely from a phone
via the GitHub website. Multi-folder projects require git or a desktop
browser to upload correctly — a phone browser can't reliably preserve
folder structure. Keeping everything in one file removes that problem
completely. We can split this into multiple files later once editing
happens from a computer or via git.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageFilter
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


class SetupQualityLabel(str, Enum):
    WEAK_NO_SETUP = "WEAK_NO_SETUP"
    LOW_QUALITY = "LOW_QUALITY"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


def label_for_score(total: float) -> SetupQualityLabel:
    if total < 50:
        return SetupQualityLabel.WEAK_NO_SETUP
    if total < 65:
        return SetupQualityLabel.LOW_QUALITY
    if total < 80:
        return SetupQualityLabel.MODERATE
    if total < 90:
        return SetupQualityLabel.STRONG
    return SetupQualityLabel.VERY_STRONG


class ConflictNote(BaseModel):
    description: str
    # "active" = a timeframe genuinely opposes the candidate direction
    # (e.g. bearish vs a bullish HTF) — this must block a hypothetical
    # setup outright. "info" = a softer gap (missing data or neutral)
    # that lowers the score but doesn't itself block a setup. Kept as
    # an explicit field rather than parsed from the text, so downstream
    # logic never has to guess intent from a human-readable string.
    severity: Literal["active", "info"] = "info"


class ConfluenceResult(BaseModel):
    """Phase 9's real output: combines the strategy engine's evidence
    (Phase 8) into ONE transparent quality score. This is a SETUP
    QUALITY SCORE, never a win probability — see the project's core
    rule against ever presenting it as one.

    HONESTY BOUNDARY: only 3 of the 7 weighted dimensions (HTF/MTF/LTF)
    have any real evidence behind them right now, because only one
    strategy module exists (Phase 8). Liquidity, structure, and
    risk/reward are held at 0 with an explanation rather than estimated
    — which caps the maximum achievable score at 60/100 (LOW QUALITY)
    until more modules and the level-calculation engine (Phase 10)
    exist. That ceiling is intentional, not a bug: the system should not
    be able to claim high confidence from one evidence source."""

    session_id: str
    computable: bool
    reason_if_not_computable: Optional[str] = None
    candidate_direction: Optional[Bias] = None
    quality_breakdown: Optional[SetupQualityBreakdown] = None
    quality_label: Optional[SetupQualityLabel] = None
    agreements: List[str] = []
    conflicts: List[ConflictNote] = []
    unscored_dimension_notes: List[str] = []


def compute_confluence(session: AnalysisSession) -> ConfluenceResult:
    """Combines this session's Phase 8 StrategyResults into a
    ConfluenceResult. Reuses cached data only — makes no new AI calls.

    Rule set (documented here because the whole point of a quality
    score is that its derivation is inspectable, not a black box):
      1. HTF sets the candidate direction. If HTF's reading is missing,
         insufficient, or neutral, NO candidate direction can be
         established and no score is computed — matches the project
         rule that HTF should determine broad context, not be
         overridden or guessed around.
      2. HTF alignment (20 pts): full weight once HTF establishes the
         candidate direction (it is the anchor).
      3. MTF alignment (20 pts) / LTF confirmation (20 pts): full
         weight only if that timeframe's reading is present, not
         insufficient, and MATCHES the candidate direction. Zero
         otherwise — including when it's simply neutral or missing,
         not only when it actively conflicts.
      4. Liquidity / Structure / Location / Risk-Reward (10 pts each,
         40 total): always 0 right now. No module produces this
         evidence yet (Phase 8 shipped exactly one module), so
         estimating these would mean fabricating a number.
      5. Any timeframe that actively conflicts with the candidate
         direction (determinate but opposite) is recorded as a
         ConflictNote, never hidden.
    """

    results = run_strategy_engine(session)
    by_role = {r.timeframe_role: r for r in results}
    htf_r, mtf_r, ltf_r = by_role.get(TimeframeRole.HTF), by_role.get(TimeframeRole.MTF), by_role.get(TimeframeRole.LTF)

    if htf_r is None or htf_r.insufficient_data:
        return ConfluenceResult(
            session_id=session.session_id, computable=False,
            reason_if_not_computable="HTF has no usable reading yet — run Compare Timeframes first. "
            "HTF sets the candidate direction, so no score can be computed without it.",
        )
    if htf_r.bias == Bias.NEUTRAL:
        return ConfluenceResult(
            session_id=session.session_id, computable=False,
            reason_if_not_computable="HTF's visual impression is neutral — there is no directional "
            "context to score against. This isn't a bug: an unclear higher timeframe honestly means "
            "no confident setup direction can be proposed yet.",
        )

    direction = htf_r.bias
    breakdown = SetupQualityBreakdown()
    breakdown.htf_alignment = 20.0
    agreements = [f"HTF establishes a {direction.value} candidate direction."]
    conflicts: List[ConflictNote] = []

    for label, weight_field, result in [("MTF", "mtf_alignment", mtf_r), ("LTF", "ltf_confirmation", ltf_r)]:
        if result is None or result.insufficient_data:
            conflicts.append(ConflictNote(
                description=f"{label} has no usable reading — treated as non-confirming, not ignored.",
                severity="info",
            ))
            continue
        if result.bias == direction:
            setattr(breakdown, weight_field, 20.0)
            agreements.append(f"{label} agrees with the {direction.value} candidate direction.")
        elif result.bias == Bias.NEUTRAL:
            conflicts.append(ConflictNote(
                description=f"{label} is neutral — neither confirms nor conflicts with {direction.value}.",
                severity="info",
            ))
        else:
            conflicts.append(ConflictNote(
                description=f"{label} shows {result.bias.value}, which conflicts with the {direction.value} "
                "candidate direction from HTF. Per this project's rule, a lower timeframe never automatically "
                "overrides a higher one — this conflict is reflected in a lower score, not resolved for you.",
                severity="active",
            ))

    unscored_notes = [
        "Liquidity: 0/10 — no liquidity-sweep or liquidity-pool detection module exists yet.",
        "Structure: 0/10 — no BOS/CHoCH/order-block detection module exists yet.",
        "Location: 0/10 — no premium/discount or key-zone-location module exists yet.",
        "Risk/Reward: 0/10 — no entry/stop/target levels exist yet (that's Phase 10).",
    ]

    breakdown.total = (
        breakdown.htf_alignment + breakdown.mtf_alignment + breakdown.ltf_confirmation
        + breakdown.liquidity + breakdown.structure + breakdown.location + breakdown.risk_reward
    )

    return ConfluenceResult(
        session_id=session.session_id, computable=True,
        candidate_direction=direction, quality_breakdown=breakdown,
        quality_label=label_for_score(breakdown.total),
        agreements=agreements, conflicts=conflicts, unscored_dimension_notes=unscored_notes,
    )


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
    image_ref: str  # points into the in-memory chart image store, see SECTION 3C
    uploaded_at: str  # ISO 8601
    quality: ChartQuality = ChartQuality.UNKNOWN
    quality_notes: List[str] = []  # human-readable reasons behind the quality label
    width_px: Optional[int] = None
    height_px: Optional[int] = None


class AnalysisStatus(str, Enum):
    WAITING_FOR_CHARTS = "WAITING_FOR_CHARTS"
    READY = "READY"
    ANALYZING = "ANALYZING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class AnalysisSession(BaseModel):
    """One session = exactly one instrument + up to three charts
    (HTF/MTF/LTF). Sessions are the boundary that prevents charts from
    different instruments ever being combined — enforced by the session
    store below (session_id is always required to attach a chart, and
    the instrument is fixed at creation and never changes)."""

    session_id: str
    instrument_id: str
    instrument_symbol: str  # denormalized for display, so the UI never
                             # has to re-look-up the instrument just to
                             # show a session's name
    created_at: str  # ISO 8601
    htf_chart_id: Optional[str] = None
    mtf_chart_id: Optional[str] = None
    ltf_chart_id: Optional[str] = None
    status: AnalysisStatus = AnalysisStatus.WAITING_FOR_CHARTS


def required_charts_present(session: AnalysisSession) -> bool:
    return session.htf_chart_id is not None and session.mtf_chart_id is not None and session.ltf_chart_id is not None


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
    """Structured, provider-agnostic output of chart vision. Deliberately
    carries NO trading conclusion (no bias, no bullish/bearish) — only
    what the model could visibly read off the image. Turning this into
    trading evidence is the strategy engine's job (a later phase)."""

    success: bool
    detected_instrument_symbol: Optional[str] = None
    detected_timeframe: Optional[str] = None
    visible_candle_count: Optional[int] = None
    raw_observations: List[str] = []
    failure_reason: Optional[str] = None


class ExplanationResult(BaseModel):
    success: bool
    text: Optional[str] = None
    failure_reason: Optional[str] = None


class VisualBiasResult(BaseModel):
    """The output of a per-timeframe visual read (Phase 7). Deliberately
    NOT a StrategyResult (no evidence list, no invalidation, no quality
    contribution) — those only exist once real strategy modules are
    built (Phase 8). This is one level below that: just "what does the
    price shape on THIS chart look like," self-reported by the AI as an
    impression, never as a trading signal."""

    success: bool
    visual_bias: Optional[Bias] = None  # None = AI couldn't tell / unclear
    reasoning: Optional[str] = None
    failure_reason: Optional[str] = None


class NamedZone(BaseModel):
    """A price zone the AI says it can see (order block, FVG, supply/
    demand zone, etc). low/high are optional on purpose — if the AI
    isn't confident enough to read exact numbers off the price axis, it
    should describe the zone in `note` instead of inventing coordinates.
    This is still an AI visual read, not a geometrically computed zone."""

    label: str
    low: Optional[float] = None
    high: Optional[float] = None
    note: Optional[str] = None


class QuickScanResult(BaseModel):
    """Single-chart 'AI Chart Scanner' output. Unlike PriceLevelReading
    (Phase 10), this DOES include an AI-proposed entry/SL/TP — that's
    the explicit trade-off of one-chart analysis (there's no second or
    third timeframe to independently cross-check against). To keep this
    honest anyway:
      - image_readable distinguishes "bad screenshot" from "readable
        chart, no clean trade" — these are NOT the same failure.
      - risk_reward_* fields are always RECOMPUTED by our own code from
        the entry/stop/TP numbers, never trusted as an AI-stated ratio.
      - entry/stop/TP are validated for basic direction sanity
        (validate_and_normalize_quick_scan, below) and discarded rather
        than shown if they don't make arithmetic sense.
      - confidence_score is an evidence score, explicitly never
        presented as a win probability.
      - EVERYTHING in the SMC / supply-demand / CRT fields below is
        still the AI's own visual judgment — NOT an independently coded
        geometric detector. A real coded BOS/FVG/order-block algorithm
        would need actual OHLC price data, which this app doesn't have
        (only a chart screenshot). These fields exist to make the AI's
        answer organized and labeled by concept, not to claim a
        different, more rigorous kind of analysis is happening.
    """

    success: bool
    failure_reason: Optional[str] = None

    image_readable: bool = True
    image_quality_note: Optional[str] = None

    detected_instrument_symbol: Optional[str] = None
    detected_timeframe: Optional[str] = None
    current_price: Optional[float] = None

    market_structure: Optional[str] = None
    trend: Optional[str] = None
    support_levels: List[float] = []
    resistance_levels: List[float] = []
    patterns: List[str] = []
    price_action_notes: List[str] = []
    strategy_signals: List[str] = []
    conflicts: List[str] = []

    # --- Smart Money Concepts (AI-identified) ---
    smc_structure_note: Optional[str] = None  # e.g. "Bullish BOS above prior swing high"
    smc_liquidity_notes: List[str] = []  # sweeps, equal highs/lows, stop-runs
    order_blocks: List[NamedZone] = []
    fair_value_gaps: List[NamedZone] = []
    premium_discount_zone: Optional[str] = None  # "premium" / "discount" / "equilibrium" / null

    # --- Supply & Demand (AI-identified) ---
    supply_zones: List[NamedZone] = []
    demand_zones: List[NamedZone] = []

    # --- CRT: Candle Range Theory (AI-identified) ---
    crt_phase: Optional[str] = None  # "accumulation" / "manipulation" / "distribution" / null
    crt_notes: Optional[str] = None

    bias: Optional[Bias] = None  # None + no_trade_reason set = a real NO TRADE result
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None

    # Always recomputed in code from the levels above — never trust an
    # AI-stated ratio directly. None if levels are missing/inconsistent.
    risk_reward_tp1: Optional[float] = None
    risk_reward_tp2: Optional[float] = None

    confidence_score: Optional[float] = None  # 0-100 evidence score, NOT a win probability
    confluence_level: Optional[str] = None  # "HIGH" / "MODERATE" / "LOW" / "NO_TRADE"

    invalidation: Optional[str] = None
    reasoning: Optional[str] = None
    no_trade_reason: Optional[str] = None

    levels_discarded_reason: Optional[str] = None  # set if AI's numbers failed our sanity check


def validate_and_normalize_quick_scan(result: QuickScanResult) -> QuickScanResult:
    """Deterministic, code-only safety net for Quick Scan's AI-proposed
    levels — 100% Python, zero AI calls in this function. This is the
    honesty backstop for the one thing Quick Scan does differently from
    the Multi-Timeframe path (letting the AI propose entry/SL/TP
    directly): it never trusts those numbers blindly.
      - Risk/reward is always RECOMPUTED here from entry/stop/TP, never
        taken from anything the AI stated as a ratio.
      - If entry or stop is missing, or the stop is on the wrong side
        of entry for the claimed direction, or TP1 is on the wrong
        side, or risk/reward is below the same 1:1.2 floor used
        elsewhere in this app — the levels are discarded and the result
        is downgraded to NO TRADE, with the reason shown, not hidden.
    """
    if not result.success or not result.image_readable or result.bias is None:
        return result

    if result.entry_zone_low is not None and result.entry_zone_high is not None:
        entry_ref = (result.entry_zone_low + result.entry_zone_high) / 2
    elif result.entry_zone_low is not None:
        entry_ref = result.entry_zone_low
    elif result.entry_zone_high is not None:
        entry_ref = result.entry_zone_high
    else:
        entry_ref = result.current_price

    def discard(reason: str) -> QuickScanResult:
        data = result.model_dump()
        data.update({
            "bias": None, "no_trade_reason": reason, "levels_discarded_reason": reason,
            "entry_zone_low": None, "entry_zone_high": None, "stop_loss": None,
            "take_profit_1": None, "take_profit_2": None, "take_profit_3": None,
            "risk_reward_tp1": None, "risk_reward_tp2": None, "confluence_level": "NO_TRADE",
        })
        return QuickScanResult(**data)

    if entry_ref is None or result.stop_loss is None:
        return discard("AI did not provide a usable entry and stop level — discarded rather than shown.")

    if result.bias == Bias.BULLISH:
        if result.stop_loss >= entry_ref:
            return discard("AI-proposed stop was not below entry for a long — directionally inconsistent, discarded.")
        if result.take_profit_1 is not None and result.take_profit_1 <= entry_ref:
            return discard("AI-proposed TP1 was not above entry for a long — directionally inconsistent, discarded.")
        risk = entry_ref - result.stop_loss
    else:  # BEARISH
        if result.stop_loss <= entry_ref:
            return discard("AI-proposed stop was not above entry for a short — directionally inconsistent, discarded.")
        if result.take_profit_1 is not None and result.take_profit_1 >= entry_ref:
            return discard("AI-proposed TP1 was not below entry for a short — directionally inconsistent, discarded.")
        risk = result.stop_loss - entry_ref

    if risk <= 0:
        return discard("Calculated risk was zero or negative — invalid setup, discarded.")

    rr1 = abs(result.take_profit_1 - entry_ref) / risk if result.take_profit_1 is not None else None
    rr2 = abs(result.take_profit_2 - entry_ref) / risk if result.take_profit_2 is not None else None

    if rr1 is not None and rr1 < 1.2:
        return discard(f"Risk/reward at TP1 ({rr1:.2f}) is below this app's 1:1.2 minimum — not a usable setup.")

    data = result.model_dump()
    data["risk_reward_tp1"] = round(rr1, 2) if rr1 is not None else None
    data["risk_reward_tp2"] = round(rr2, 2) if rr2 is not None else None
    return QuickScanResult(**data)


class AIProvider(ABC):
    @abstractmethod
    def get_capabilities(self) -> AIProviderCapabilities: ...

    @abstractmethod
    def interpret_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> ChartInterpretationResult: ...

    @abstractmethod
    def analyze_visual_bias(
        self, image_bytes: bytes, role: "TimeframeRole", instrument_hint: Optional[str] = None
    ) -> VisualBiasResult: ...

    @abstractmethod
    def extract_price_levels(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> PriceLevelReading: ...

    @abstractmethod
    def quick_scan_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None, style_hint: Optional[str] = None
    ) -> QuickScanResult: ...

    @abstractmethod
    def explain(self, context: Dict[str, Any], instruction: str) -> ExplanationResult: ...


class NoOpAIProvider(AIProvider):
    """Default provider. Honestly reports that no AI is connected rather
    than pretending to analyze a chart. Used whenever AI_PROVIDER is
    unset/"none", or when "gemini" is selected but no API key was found —
    never silently falls back to fake success."""

    def __init__(self, reason: str = "No AI provider is configured yet."):
        self._reason = reason

    def get_capabilities(self) -> AIProviderCapabilities:
        return AIProviderCapabilities(
            supports_vision=False,
            supports_text=False,
            provider_name="None configured",
        )

    def analyze_visual_bias(
        self, image_bytes: bytes, role: "TimeframeRole", instrument_hint: Optional[str] = None
    ) -> VisualBiasResult:
        return VisualBiasResult(success=False, failure_reason=self._reason)

    def extract_price_levels(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> PriceLevelReading:
        return PriceLevelReading(success=False, failure_reason=self._reason)

    def quick_scan_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None, style_hint: Optional[str] = None
    ) -> QuickScanResult:
        return QuickScanResult(success=False, failure_reason=self._reason)

    def interpret_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> ChartInterpretationResult:
        return ChartInterpretationResult(success=False, failure_reason=self._reason)

    def explain(self, context: Dict[str, Any], instruction: str) -> ExplanationResult:
        return ExplanationResult(success=False, failure_reason=self._reason)


class GeminiAIProvider(AIProvider):
    """Google Gemini free tier (Google AI Studio), chosen for Phase 6
    because its free tier treats vision as a core, non-preview
    capability with more generous daily limits than the alternatives
    (see Settings for a plain-language summary).

    HONESTY NOTES, surfaced to the user rather than hidden:
      - The free tier is rate-limited (~10 requests/minute, ~1,500/day
        at time of writing) and Google may use free-tier prompts/images
        to improve their products — this is disclosed in Settings.
      - The model is prompted to report ONLY visible observations, never
        a trading conclusion — that separation is what keeps "AI
        explains, engine calculates" real instead of just a slogan.
      - Every failure mode (missing key, rate limit, bad/empty
        response, timeout, malformed JSON) returns success=False with a
        specific, honest reason — never a fabricated result.
    """

    # Pinned to a specific STABLE (non-preview) model rather than the
    # "gemini-flash-latest" alias. That alias currently resolves to
    # gemini-3-flash-preview, which returns frequent 503 "high demand"
    # errors on the free tier — a real, widely-reported issue (Google's
    # own developer forum has open threads on it), not a fluke.
    # gemini-3.1-flash-lite is GA/stable, vision-capable, and has a
    # comparable free tier (30 req/min, 1,500/day, no card). Trade-off:
    # unlike an auto-updating alias, this pinned name will eventually
    # need a manual update if Google deprecates it — an acceptable cost
    # for far fewer 503s today.
    MODEL = "gemini-3.1-flash-lite"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_capabilities(self) -> AIProviderCapabilities:
        return AIProviderCapabilities(
            supports_vision=True,
            supports_text=True,
            provider_name=f"Google Gemini (free tier · {self.MODEL})",
            is_free_tier=True,
        )

    def _mime_type_for(self, image_bytes: bytes) -> str:
        try:
            fmt = (Image.open(io.BytesIO(image_bytes)).format or "PNG").lower()
        except Exception:
            fmt = "png"
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in ("png", "jpeg", "webp"):
            fmt = "png"
        return f"image/{fmt}"

    def _call_gemini_vision(
        self, prompt: str, image_bytes: Optional[bytes] = None, max_output_tokens: int = 700
    ) -> Tuple[Optional[str], Optional[str]]:
        """Shared request/error-handling for any Gemini call — with or
        without an attached image (image_bytes=None makes this a
        text-only call, used for the Quick Scan JSON-repair retry).
        Automatically retries up to 2 extra times, with a short delay,
        ONLY on 503/502/504 (server-side overload/unavailable) — never
        on 429 (rate limit) or 400 (bad request), where retrying
        immediately would just waste quota or repeat the same error.
        Returns (raw_text, failure_reason) — exactly one is non-None."""

        parts = [{"text": prompt}]
        if image_bytes is not None:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            parts.append({"inline_data": {"mime_type": self._mime_type_for(image_bytes), "data": b64}})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_output_tokens},
        }

        max_attempts = 3
        last_5xx_status = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(url, params={"key": self._api_key}, json=payload, timeout=30)
            except requests.exceptions.Timeout:
                return None, "Gemini request timed out after 30 seconds."
            except requests.exceptions.RequestException as e:
                return None, f"Network error calling Gemini: {e}"

            if resp.status_code == 429:
                return None, "Gemini's free-tier rate limit was hit. Wait a minute and try again."
            if resp.status_code == 400:
                return None, "Gemini rejected this request (often an invalid API key). Check Settings/Secrets."
            if resp.status_code in (500, 502, 503, 504):
                last_5xx_status = resp.status_code
                if attempt < max_attempts:
                    time.sleep(2 * attempt)  # 2s, then 4s
                    continue
                return None, (
                    f"Gemini's servers are still overloaded or unavailable after {max_attempts} tries "
                    f"(status {last_5xx_status}). This is on Google's side, not a bug here — it usually "
                    "clears up within a few minutes; try again shortly."
                )
            if resp.status_code != 200:
                return None, f"Gemini API returned an unexpected status ({resp.status_code})."

            try:
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    return None, "Gemini returned no result for this image."
                text = candidates[0]["content"]["parts"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.lower().startswith("json"):
                        text = text[4:]
                return text, None
            except (KeyError, IndexError) as e:
                return None, f"Gemini's response wasn't in the expected format ({e})."

        return None, "Unexpected retry loop exit."  # unreachable, kept for type-safety

    def interpret_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> ChartInterpretationResult:
        prompt = (
            "You are looking at a screenshot of a financial trading chart. "
            "Report ONLY what is visibly present in the image. Do NOT give "
            "any trading advice, opinion, prediction, or bullish/bearish "
            "conclusion — that is out of scope for this task. "
            "Respond with ONLY a raw JSON object (no markdown fences, no "
            "extra text) with exactly these keys: "
            '{"detected_instrument_symbol": string or null, '
            '"detected_timeframe": string or null, '
            '"visible_candle_count_estimate": integer or null, '
            '"observations": array of short strings describing only what is '
            "visibly present — e.g. chart type, axis labels visible, general "
            "shape of the price line/candles, colors used, any UI text "
            "visible. Do not describe trend direction as advice, only as a "
            "visual fact if clearly labeled.}"
        )
        if instrument_hint:
            prompt += (
                f' The user believes this chart is for instrument "{instrument_hint}" — '
                "confirm or correct that only if you can actually read a symbol in the image."
            )

        text, failure = self._call_gemini_vision(prompt, image_bytes)
        if failure:
            return ChartInterpretationResult(success=False, failure_reason=failure)

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            return ChartInterpretationResult(
                success=False, failure_reason=f"Gemini's response wasn't valid JSON ({e})."
            )

        return ChartInterpretationResult(
            success=True,
            detected_instrument_symbol=parsed.get("detected_instrument_symbol"),
            detected_timeframe=parsed.get("detected_timeframe"),
            visible_candle_count=parsed.get("visible_candle_count_estimate"),
            raw_observations=list(parsed.get("observations") or []),
        )

    def analyze_visual_bias(
        self, image_bytes: bytes, role: "TimeframeRole", instrument_hint: Optional[str] = None
    ) -> VisualBiasResult:
        """Phase 7: a role-aware read of ONE chart's visible price shape.
        Still not a trading signal — see VisualBiasResult's docstring.
        Combining these across HTF/MTF/LTF into agreement/conflict happens
        in compute_timeframe_comparison(), not here."""

        role_context = {
            "HTF": "This is the HIGHER TIME FRAME chart — used for broad context, not fine detail.",
            "MTF": "This is the MIDDLE TIME FRAME chart — developing structure and setup location.",
            "LTF": "This is the LOWER TIME FRAME chart — short-term execution detail.",
        }.get(role.value, "")

        prompt = (
            f"You are looking at a screenshot of a financial trading chart. {role_context} "
            "Based ONLY on the visible shape of the price movement in THIS image (e.g. a "
            "general pattern of higher highs/higher lows, lower highs/lower lows, or a flat "
            "sideways range), describe the general visual impression of the price direction "
            "shown. This is a description of a visual shape, NOT trading advice and NOT a "
            "prediction of what happens next. "
            "Respond with ONLY a raw JSON object (no markdown fences, no extra text) with "
            "exactly these keys: "
            '{"visual_bias": one of "bullish", "bearish", "neutral", or null if the visible '
            "price action doesn't clearly show any of those, "
            '"reasoning": a short one-sentence description referencing only what is visibly '
            "plotted (e.g. 'price makes higher highs and higher lows across the visible "
            "candles').}"
        )
        if instrument_hint:
            prompt += f' The user believes this chart is for instrument "{instrument_hint}".'

        text, failure = self._call_gemini_vision(prompt, image_bytes)
        if failure:
            return VisualBiasResult(success=False, failure_reason=failure)

        try:
            parsed = json.loads(text)
            raw_bias = parsed.get("visual_bias")
            bias = Bias(raw_bias) if raw_bias in ("bullish", "bearish", "neutral") else None
        except (json.JSONDecodeError, ValueError) as e:
            return VisualBiasResult(success=False, failure_reason=f"Gemini's response wasn't valid JSON ({e}).")

        return VisualBiasResult(success=True, visual_bias=bias, reasoning=parsed.get("reasoning"))

    def extract_price_levels(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> PriceLevelReading:
        """Phase 10: reads REAL numbers directly off the chart's visible
        price axis/labels. This is OCR-style reading of what's printed
        on screen, not calculation or estimation — the level-calculation
        engine (calculate_hypothetical_setup, below) does all the actual
        math in plain Python, never asking the AI to compute anything."""

        prompt = (
            "You are looking at a screenshot of a financial trading chart. Read ONLY "
            "numbers that are actually visible as printed price labels or that you can "
            "confidently place using the visible price axis scale. Identify: "
            "(1) the current/most recent price (often a highlighted price label, usually "
            "on the right side), "
            "(2) the price level of the most recent significant swing high visible near "
            "the right/current side of the chart, "
            "(3) the price level of the most recent significant swing low visible near "
            "the right/current side of the chart. "
            "If you cannot confidently determine any of these from what's actually "
            "visible, return null for that field — do NOT guess or estimate. "
            "Respond with ONLY a raw JSON object (no markdown fences, no extra text) with "
            "exactly these keys: "
            '{"current_price": number or null, "recent_swing_high": number or null, '
            '"recent_swing_low": number or null, "confidence_notes": a short sentence '
            "explaining what you could/couldn't read.}"
        )
        if instrument_hint:
            prompt += f' The user believes this chart is for instrument "{instrument_hint}".'

        text, failure = self._call_gemini_vision(prompt, image_bytes)
        if failure:
            return PriceLevelReading(success=False, failure_reason=failure)

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            return PriceLevelReading(success=False, failure_reason=f"Gemini's response wasn't valid JSON ({e}).")

        def _as_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return PriceLevelReading(
            success=True,
            current_price=_as_float(parsed.get("current_price")),
            recent_swing_high=_as_float(parsed.get("recent_swing_high")),
            recent_swing_low=_as_float(parsed.get("recent_swing_low")),
            confidence_notes=parsed.get("confidence_notes"),
        )

    QUICK_SCAN_SCHEMA_KEYS = (
        '{"image_readable": true or false, '
        '"image_quality_note": string or null, '
        '"detected_instrument_symbol": string or null, '
        '"detected_timeframe": string or null, '
        '"current_price": number or null, '
        '"market_structure": short string or null, '
        '"trend": one of "bullish","bearish","neutral","ranging", or null, '
        '"support_levels": array of numbers (empty if none confidently visible), '
        '"resistance_levels": array of numbers (empty if none confidently visible), '
        '"patterns": array of short strings naming any visible chart patterns, '
        '"price_action_notes": array of short strings, '
        '"strategy_signals": array of short strings naming which technical concepts '
        "support the thesis (e.g. \"break of structure\", \"liquidity sweep\"), only if "
        "genuinely visible, "
        '"conflicts": array of short strings describing anything that WEAKENS the thesis '
        "(e.g. nearby resistance, weak momentum), "
        '"smc_structure_note": a short Smart Money Concepts structure read (e.g. '
        '"Bullish BOS confirmed above the prior swing high"), or null if not clearly visible, '
        '"smc_liquidity_notes": array of short strings on liquidity sweeps, equal highs/lows, '
        "or stop-runs, only if genuinely visible, "
        '"order_blocks": array of objects {"label": string, "low": number or null, '
        '"high": number or null, "note": string or null} for any order blocks visible — '
        "leave low/high null rather than guessing if you can't read exact levels, "
        '"fair_value_gaps": array of objects in the same {label, low, high, note} shape '
        "for any fair value gaps / imbalances visible, "
        '"premium_discount_zone": one of "premium","discount","equilibrium", or null, '
        '"supply_zones": array of objects in the same {label, low, high, note} shape, '
        '"demand_zones": array of objects in the same {label, low, high, note} shape, '
        '"crt_phase": Candle Range Theory phase — one of "accumulation","manipulation",'
        '"distribution", or null if a defined range candle isn\'t clearly identifiable, '
        '"crt_notes": short string explaining the CRT read, or null, '
        '"bias": one of "bullish","bearish", or null (null means NO TRADE), '
        '"entry_zone_low": number or null, "entry_zone_high": number or null, '
        '"stop_loss": number or null, '
        '"take_profit_1": number or null, "take_profit_2": number or null, "take_profit_3": number or null, '
        '"confidence_score": number 0-100 (evidence-based, not a win probability) or null, '
        '"confluence_level": one of "HIGH","MODERATE","LOW","NO_TRADE", '
        '"invalidation": short string or null, '
        '"reasoning": short paragraph explaining the thesis in plain language, '
        '"no_trade_reason": string or null (REQUIRED if bias is null)}'
    )

    def quick_scan_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None, style_hint: Optional[str] = None
    ) -> QuickScanResult:
        """Single-chart 'AI Chart Scanner' analysis — one integrated call
        that both reads the chart AND proposes a structured setup. This
        is different from the Multi-Timeframe path's strict separation
        (AI reads numbers only, code does all math) — here the AI
        proposes entry/SL/TP directly, because with only one chart
        there's no second/third timeframe to independently derive
        levels from. The safety net is entirely downstream, in
        validate_and_normalize_quick_scan(): risk/reward is always
        recomputed by our own code, and directionally inconsistent
        levels are discarded rather than shown."""

        prompt = (
            "You are a technical chart-analysis vision model. Analyze ONLY evidence "
            "visible in this single chart screenshot. Do NOT invent prices, indicators, "
            "market structure, or patterns that are not visible. Identify uncertainty "
            "explicitly rather than guessing. Do NOT force a trade — if there is no "
            "coherent, evidence-supported setup, set bias to null and explain why in "
            "no_trade_reason; this is a normal, expected, and equally valid result. "
            "If the image itself is too blurry, too zoomed, obstructed, or doesn't show "
            "enough candles/price scale to analyze at all, set image_readable to false "
            "and explain in image_quality_note — do not attempt analysis on an unreadable "
            "image. Only include entry/stop/take-profit numbers if there is a visible "
            "price scale you can confidently read; otherwise leave them null. Separate "
            "supporting evidence (strategy_signals) from things that weaken the thesis "
            "(conflicts) explicitly. Never claim certainty — confidence_score is an "
            "evidence-based score, not a probability of winning.\n\n"
            "In addition to general price action, specifically look for and report on "
            "(leaving any field null if genuinely not visible — do not force these to "
            "apply if the chart doesn't support them):\n"
            "- SMART MONEY CONCEPTS (SMC): market structure in BOS/CHoCH terms, "
            "liquidity sweeps or equal highs/lows, order blocks, fair value gaps (FVG), "
            "and whether price is trading in a premium, discount, or equilibrium zone "
            "relative to the visible range.\n"
            "- SUPPLY & DEMAND: any supply zones (areas of prior sharp selling) or "
            "demand zones (areas of prior sharp buying) that are visually identifiable, "
            "with approximate price levels only if you can confidently read them.\n"
            "- CRT (Candle Range Theory): whether a defined range candle is identifiable "
            "and, if so, which phase price appears to be in — accumulation (inside the "
            "range), manipulation (a sweep beyond the range), or distribution (a "
            "directional move away from the range). Leave crt_phase null if no clear "
            "range candle is identifiable — do not force this framework onto every chart.\n\n"
        )
        if instrument_hint:
            prompt += f'The user believes this chart is for instrument "{instrument_hint}". '
        if style_hint:
            prompt += f"The user's preferred trading style is {style_hint} — weigh your analysis toward what matters for that style, but do not fabricate evidence to fit it. "
        prompt += (
            "\n\nRespond with ONLY a raw JSON object (no markdown fences, no extra text) "
            f"matching EXACTLY this schema: {self.QUICK_SCAN_SCHEMA_KEYS}"
        )

        text, failure = self._call_gemini_vision(prompt, image_bytes, max_output_tokens=2200)
        if failure:
            return QuickScanResult(success=False, failure_reason=failure)

        parsed = self._parse_json_with_repair(text, self.QUICK_SCAN_SCHEMA_KEYS)
        if parsed is None:
            return QuickScanResult(
                success=False,
                failure_reason="Gemini's response wasn't valid JSON, even after one repair attempt.",
            )

        def _as_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _as_float_list(v):
            if not isinstance(v, list):
                return []
            out = []
            for x in v:
                f = _as_float(x)
                if f is not None:
                    out.append(f)
            return out

        def _as_zone_list(v):
            """Tolerant parsing — a malformed zone entry is skipped, never
            allowed to crash the whole scan or silently invent numbers."""
            if not isinstance(v, list):
                return []
            out = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                label = item.get("label")
                if not label:
                    continue
                out.append(NamedZone(
                    label=str(label),
                    low=_as_float(item.get("low")),
                    high=_as_float(item.get("high")),
                    note=item.get("note"),
                ))
            return out

        raw_bias = parsed.get("bias")
        bias = Bias(raw_bias) if raw_bias in ("bullish", "bearish") else None

        result = QuickScanResult(
            success=True,
            image_readable=bool(parsed.get("image_readable", True)),
            image_quality_note=parsed.get("image_quality_note"),
            detected_instrument_symbol=parsed.get("detected_instrument_symbol"),
            detected_timeframe=parsed.get("detected_timeframe"),
            current_price=_as_float(parsed.get("current_price")),
            market_structure=parsed.get("market_structure"),
            trend=parsed.get("trend"),
            support_levels=_as_float_list(parsed.get("support_levels")),
            resistance_levels=_as_float_list(parsed.get("resistance_levels")),
            patterns=list(parsed.get("patterns") or []),
            price_action_notes=list(parsed.get("price_action_notes") or []),
            strategy_signals=list(parsed.get("strategy_signals") or []),
            conflicts=list(parsed.get("conflicts") or []),
            smc_structure_note=parsed.get("smc_structure_note"),
            smc_liquidity_notes=list(parsed.get("smc_liquidity_notes") or []),
            order_blocks=_as_zone_list(parsed.get("order_blocks")),
            fair_value_gaps=_as_zone_list(parsed.get("fair_value_gaps")),
            premium_discount_zone=parsed.get("premium_discount_zone"),
            supply_zones=_as_zone_list(parsed.get("supply_zones")),
            demand_zones=_as_zone_list(parsed.get("demand_zones")),
            crt_phase=parsed.get("crt_phase"),
            crt_notes=parsed.get("crt_notes"),
            bias=bias,
            entry_zone_low=_as_float(parsed.get("entry_zone_low")),
            entry_zone_high=_as_float(parsed.get("entry_zone_high")),
            stop_loss=_as_float(parsed.get("stop_loss")),
            take_profit_1=_as_float(parsed.get("take_profit_1")),
            take_profit_2=_as_float(parsed.get("take_profit_2")),
            take_profit_3=_as_float(parsed.get("take_profit_3")),
            confidence_score=_as_float(parsed.get("confidence_score")),
            confluence_level=parsed.get("confluence_level"),
            invalidation=parsed.get("invalidation"),
            reasoning=parsed.get("reasoning"),
            no_trade_reason=parsed.get("no_trade_reason"),
        )
        return validate_and_normalize_quick_scan(result)

    def _parse_json_with_repair(self, text: str, schema_keys: str) -> Optional[dict]:
        """Tries to parse `text` as JSON. If that fails, makes ONE
        additional text-only Gemini call asking it to repair its own
        output into valid JSON, per the project's fallback rule (retry
        once, then fail cleanly — never fabricate on repeated failure)."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        repair_prompt = (
            "The following text was supposed to be a single raw JSON object matching "
            f"this schema: {schema_keys}\n\nIt is not valid JSON. Return ONLY the "
            "corrected raw JSON object, no markdown fences, no explanation:\n\n" + text
        )
        repaired_text, failure = self._call_gemini_vision(repair_prompt, image_bytes=None, max_output_tokens=2200)
        if failure or repaired_text is None:
            return None
        try:
            return json.loads(repaired_text)
        except (json.JSONDecodeError, ValueError):
            return None

    def explain(self, context: Dict[str, Any], instruction: str) -> ExplanationResult:
        try:
            prompt = (
                f"{instruction}\n\nUse ONLY the structured data below — never invent "
                f"numbers or facts not present in it:\n{json.dumps(context, indent=2, default=str)}"
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
            }
            resp = requests.post(url, params={"key": self._api_key}, json=payload, timeout=30)
        except requests.exceptions.Timeout:
            return ExplanationResult(success=False, failure_reason="Gemini request timed out after 30 seconds.")
        except requests.exceptions.RequestException as e:
            return ExplanationResult(success=False, failure_reason=f"Network error calling Gemini: {e}")

        if resp.status_code == 429:
            return ExplanationResult(success=False, failure_reason="Gemini's free-tier rate limit was hit. Try again shortly.")
        if resp.status_code in (500, 502, 503, 504):
            return ExplanationResult(
                success=False,
                failure_reason=(
                    f"Gemini's servers are temporarily overloaded or unavailable (status {resp.status_code}). "
                    "Wait 30-60 seconds and try again."
                ),
            )
        if resp.status_code != 200:
            return ExplanationResult(success=False, failure_reason=f"Gemini API returned an unexpected status ({resp.status_code}).")

        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            return ExplanationResult(success=False, failure_reason=f"Gemini's response wasn't in the expected format ({e}).")

        return ExplanationResult(success=True, text=text)


def _gemini_api_key() -> Optional[str]:
    return os.environ.get("AI_PROVIDER_API_KEY") or os.environ.get("GEMINI_API_KEY")


def get_configured_ai_provider() -> AIProvider:
    selected = os.environ.get("AI_PROVIDER", "none").strip().lower()
    if selected == "gemini":
        api_key = _gemini_api_key()
        if api_key:
            return GeminiAIProvider(api_key)
        return NoOpAIProvider(
            'AI_PROVIDER is set to "gemini" but no API key was found. '
            "Add AI_PROVIDER_API_KEY in Streamlit Cloud Secrets."
        )
    return NoOpAIProvider()


def ai_setup_warning() -> Optional[str]:
    """Surfaced in Settings — never lets a misconfiguration pass silently."""
    selected = os.environ.get("AI_PROVIDER", "none").strip().lower()
    if selected == "gemini" and not _gemini_api_key():
        return (
            'AI_PROVIDER is set to "gemini" but no AI_PROVIDER_API_KEY (or GEMINI_API_KEY) '
            "was found. Vision analysis will not work until this is added in Secrets."
        )
    return None


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
# SECTION 3: STRATEGY REGISTRY (Phase 8)
# (equivalent to tradelens/engines/strategy_registry.py + one real module)
#
# Shipped EMPTY through Phase 7 on purpose — implementing strategies
# before real evidence existed would mean fabricating analysis logic.
# Now that Phase 6 (AI vision) and Phase 7 (per-timeframe reads) exist,
# there is exactly ONE kind of real evidence available: the AI's visual
# impression of each chart. So there is exactly ONE real module here —
# not a stand-in for 70, an honest reflection of what's actually
# computable right now. Named/technical strategies (BOS, CHoCH, FVG,
# order blocks, RSI, etc.) need real structured price/indicator data
# this app doesn't have yet — adding them now would mean the AI
# inventing structure that isn't verifiably there.
# =====================================================================


class StrategyModuleInfo(BaseModel):
    strategy_id: str
    name: str
    description: str
    layers_covered: List[EvidenceLayer]


STRATEGY_REGISTRY: List[StrategyModuleInfo] = [
    StrategyModuleInfo(
        strategy_id="ai_visual_trend",
        name="AI Visual Trend Reading",
        description=(
            "Uses the AI's per-timeframe visual read (from the Compare Timeframes "
            "step) as directional evidence. This is a described impression of the "
            "visible price shape, not a calculated technical indicator."
        ),
        layers_covered=[EvidenceLayer.MARKET_CONTEXT, EvidenceLayer.MARKET_LOCATION, EvidenceLayer.CONFIRMATION],
    ),
]

# HTF work best answers "what's the broad context" (Layer 1), MTF best
# answers "where is price located within that context" (Layer 2), and
# LTF is the closest of the three to "is this confirmed right now"
# (Layer 4). Layers 3 (liquidity/structure) and 5 (execution) are
# intentionally left uncovered — those need real structural detection
# (order blocks, liquidity sweeps, precise entry triggers) this app
# doesn't have yet.
_ROLE_TO_LAYER: Dict[TimeframeRole, EvidenceLayer] = {
    TimeframeRole.HTF: EvidenceLayer.MARKET_CONTEXT,
    TimeframeRole.MTF: EvidenceLayer.MARKET_LOCATION,
    TimeframeRole.LTF: EvidenceLayer.CONFIRMATION,
}


def run_strategy_engine(session: AnalysisSession) -> List[StrategyResult]:
    """Runs every registered module against this session's ALREADY-CACHED
    Phase 7 bias reads — no new AI calls are made here, so running this
    doesn't spend extra free-tier quota. If a chart hasn't been read yet
    (or the read failed), the result is honestly marked
    insufficient_data=True rather than guessing a bias."""

    chart_ids = {
        TimeframeRole.HTF: session.htf_chart_id,
        TimeframeRole.MTF: session.mtf_chart_id,
        TimeframeRole.LTF: session.ltf_chart_id,
    }
    results: List[StrategyResult] = []

    for role, chart_id in chart_ids.items():
        cached = get_cached_bias(chart_id) if chart_id else None

        if cached is None:
            results.append(StrategyResult(
                strategy_id="ai_visual_trend",
                strategy_name="AI Visual Trend Reading",
                layer=_ROLE_TO_LAYER[role],
                timeframe_role=role,
                bias=Bias.NEUTRAL,  # placeholder only — insufficient_data below means "ignore this"
                insufficient_data=True,
                notes=["Not read yet — run \"Compare timeframes\" first."],
            ))
        elif not cached.success:
            results.append(StrategyResult(
                strategy_id="ai_visual_trend",
                strategy_name="AI Visual Trend Reading",
                layer=_ROLE_TO_LAYER[role],
                timeframe_role=role,
                bias=Bias.NEUTRAL,
                insufficient_data=True,
                notes=[cached.failure_reason or "AI read failed."],
            ))
        elif cached.visual_bias is None:
            results.append(StrategyResult(
                strategy_id="ai_visual_trend",
                strategy_name="AI Visual Trend Reading",
                layer=_ROLE_TO_LAYER[role],
                timeframe_role=role,
                bias=Bias.NEUTRAL,
                insufficient_data=True,
                notes=["AI could not determine a clear visual bias for this chart."],
            ))
        else:
            results.append(StrategyResult(
                strategy_id="ai_visual_trend",
                strategy_name="AI Visual Trend Reading",
                layer=_ROLE_TO_LAYER[role],
                timeframe_role=role,
                bias=cached.visual_bias,
                evidence=[cached.reasoning] if cached.reasoning else [],
                invalidation="This reading would need to be redone if the chart is replaced.",
                quality_contribution=0.0,  # this field is unused; the confluence engine (Phase 9) computes weight independently
                insufficient_data=False,
                notes=["See \"Confluence & Setup Quality\" for how this is weighted into a score."],
            ))

    return results


# =====================================================================
# SECTION 3B: SESSION STORE (Phase 3)
# (equivalent to a repository/store layer over tradelens/types/session.py)
#
# Sessions live in Streamlit's session_state, which is scoped to one
# browser tab/visit. This is real, working session tracking — it is
# just not persisted to a database yet (that's a later phase). Being
# honest about that distinction matters: closing the tab or the app
# restarting will lose sessions, and the Dashboard/Active Setups pages
# say so.
# =====================================================================


def _session_store() -> Dict[str, AnalysisSession]:
    if "sessions" not in st.session_state:
        st.session_state["sessions"] = {}
    return st.session_state["sessions"]


def create_session(instrument: Instrument) -> AnalysisSession:
    """The only way a session is created — instrument is fixed here and
    never changes for this session's lifetime. This is what makes "never
    mix charts from different instruments" enforceable in code rather
    than just a rule we hope gets followed."""

    session = AnalysisSession(
        session_id=str(uuid.uuid4())[:8],
        instrument_id=instrument.instrument_id,
        instrument_symbol=instrument.symbol,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _session_store()[session.session_id] = session
    return session


def list_sessions() -> List[AnalysisSession]:
    return sorted(_session_store().values(), key=lambda s: s.created_at, reverse=True)


def get_session(session_id: str) -> Optional[AnalysisSession]:
    return _session_store().get(session_id)


def delete_session(session_id: str) -> None:
    # Also remove this session's charts so orphaned images don't linger
    # in memory once a session is deleted.
    session = get_session(session_id)
    if session:
        for chart_id in [session.htf_chart_id, session.mtf_chart_id, session.ltf_chart_id]:
            if chart_id:
                delete_chart(chart_id)
    _session_store().pop(session_id, None)


# =====================================================================
# SECTION 3C-i: CHART QUALITY CHECK (Phase 5)
# (equivalent to tradelens/validation/chart_validation.py's image side)
#
# IMPORTANT — HONESTY BOUNDARY: this is real, deterministic image
# analysis (resolution, aspect ratio, a blur estimate, duplicate
# detection) using open-source, offline image processing. It is NOT AI
# vision — it cannot tell you what instrument or timeframe is shown, or
# whether the candles/price axis are actually readable. That requires a
# real vision model, which is Phase 6. Everything here only checks
# whether the file is technically usable, and says so explicitly.
# =====================================================================


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analyze_chart_quality(image_bytes: bytes) -> Tuple[ChartQuality, List[str], Optional[int], Optional[int]]:
    """Returns (quality_label, notes, width_px, height_px). Never raises
    — a file that can't even be opened as an image is itself a POOR
    result with an explanatory note, not a crash."""

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # forces a full decode, catching truncated/corrupt files
    except Exception:
        return (
            ChartQuality.POOR,
            ["This file couldn't be read as an image. Please upload a PNG or JPG screenshot."],
            None,
            None,
        )

    width, height = img.size
    major_issues: List[str] = []
    minor_issues: List[str] = []

    if width < 400 or height < 300:
        major_issues.append(
            f"Resolution is very low ({width}×{height}px) — chart details may not be readable."
        )
    elif width < 800 or height < 500:
        minor_issues.append(f"Resolution is a bit low ({width}×{height}px) — a bigger screenshot would help.")

    aspect = width / height if height else 0
    if aspect > 4 or aspect < 0.25:
        major_issues.append(
            "Unusual width-to-height ratio — this may be a heavily cropped strip rather than a full chart."
        )

    try:
        gray = img.convert("L")
        # Downscale for a fast, size-independent sharpness estimate.
        if width > 300:
            gray = gray.resize((300, max(1, int(300 * height / width))))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(edges, dtype=np.float32)
        # Trim the outer border — PIL's edge filter produces artifact
        # values along the image boundary (from its edge-padding
        # behavior) that aren't real content and would otherwise make a
        # completely blank image look artificially "sharp".
        if arr.shape[0] > 4 and arr.shape[1] > 4:
            arr = arr[2:-2, 2:-2]
        sharpness = float(arr.var())
    except Exception:
        sharpness = None

    if sharpness is not None:
        if sharpness < 50:
            major_issues.append("Automatic sharpness estimate is very low — image may be blurry or a blank/solid screen.")
        elif sharpness < 150:
            minor_issues.append("Automatic sharpness estimate is on the low side — image may be slightly blurry.")

    if major_issues:
        label = ChartQuality.POOR
    elif minor_issues:
        label = ChartQuality.FAIR
    else:
        label = ChartQuality.GOOD

    return label, major_issues + minor_issues, width, height


# =====================================================================
# SECTION 3C: CHART STORE (Phase 4)
# (equivalent to a repository/store layer over tradelens/types/chart.py)
#
# Like sessions, uploaded chart images live only in Streamlit's
# session_state for this browser tab — not saved to disk or a database
# yet. Each chart is always created attached to exactly one session and
# one timeframe role; there is no code path that lets a chart exist
# without both, which is what makes "never mix charts between sessions"
# an enforced fact rather than a hope.
# =====================================================================


def _chart_store() -> Dict[str, ChartUpload]:
    if "charts" not in st.session_state:
        st.session_state["charts"] = {}
    return st.session_state["charts"]


def _chart_images() -> Dict[str, bytes]:
    if "chart_images" not in st.session_state:
        st.session_state["chart_images"] = {}
    return st.session_state["chart_images"]


def attach_chart(session: AnalysisSession, role: TimeframeRole, uploaded_file) -> ChartUpload:
    """Creates a ChartUpload for the given role, runs the (non-AI) quality
    check, checks for an exact-duplicate image already in this session,
    stores its image bytes, attaches it to the session, and updates the
    session's status. This is the only function that may set a session's
    htf/mtf/ltf_chart_id — keeping that in one place is what prevents a
    chart from ever being attached to the wrong session by accident."""

    image_bytes = uploaded_file.getvalue()
    quality, notes, width, height = analyze_chart_quality(image_bytes)

    new_hash = _sha256(image_bytes)
    for other_role, other_id in [("HTF", session.htf_chart_id), ("MTF", session.mtf_chart_id), ("LTF", session.ltf_chart_id)]:
        if other_id and other_id in _chart_images():
            if _sha256(_chart_images()[other_id]) == new_hash:
                notes = [f"This looks like the exact same image already uploaded as the {other_role} chart — check you didn't upload a duplicate."] + notes

    chart_id = str(uuid.uuid4())[:8]
    chart = ChartUpload(
        chart_id=chart_id,
        session_id=session.session_id,
        instrument_id=session.instrument_id,
        timeframe_role=role,
        image_ref=f"memory:{chart_id}",
        uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        quality=quality,
        quality_notes=notes,
        width_px=width,
        height_px=height,
    )
    _chart_store()[chart_id] = chart
    _chart_images()[chart_id] = image_bytes

    if role == TimeframeRole.HTF:
        session.htf_chart_id = chart_id
    elif role == TimeframeRole.MTF:
        session.mtf_chart_id = chart_id
    else:
        session.ltf_chart_id = chart_id

    session.status = (
        AnalysisStatus.READY if required_charts_present(session) else AnalysisStatus.WAITING_FOR_CHARTS
    )
    _session_store()[session.session_id] = session
    return chart


def clear_chart_role(session: AnalysisSession, role: TimeframeRole) -> None:
    """Removes whichever chart is currently attached to this role on
    this session (e.g. so the user can replace it), without touching
    any other session."""

    chart_id = {"HTF": session.htf_chart_id, "MTF": session.mtf_chart_id, "LTF": session.ltf_chart_id}[role.value]
    if chart_id:
        delete_chart(chart_id)
    if role == TimeframeRole.HTF:
        session.htf_chart_id = None
    elif role == TimeframeRole.MTF:
        session.mtf_chart_id = None
    else:
        session.ltf_chart_id = None
    session.status = AnalysisStatus.WAITING_FOR_CHARTS
    _session_store()[session.session_id] = session


def delete_chart(chart_id: str) -> None:
    _chart_store().pop(chart_id, None)
    _chart_images().pop(chart_id, None)
    _bias_reads().pop(chart_id, None)


def get_chart_image(chart_id: str) -> Optional[bytes]:
    return _chart_images().get(chart_id)


# =====================================================================
# SECTION 3D: MULTI-TIMEFRAME COMPARISON (Phase 7)
# (equivalent to a TimeframeBias / partial ConfluenceResult — but
# WITHOUT the setup-quality score, which requires real strategy modules
# combining evidence, i.e. Phases 8-9)
#
# What this section does: runs analyze_visual_bias() once per attached
# chart (cached per chart_id so re-viewing a session doesn't re-spend
# free-tier API calls), then compares the three results. It can only
# ever say "these three visual impressions agree/disagree" — it cannot
# and does not produce a trading signal, a quality score, or a
# recommended direction.
# =====================================================================


def _bias_reads() -> Dict[str, VisualBiasResult]:
    if "bias_reads" not in st.session_state:
        st.session_state["bias_reads"] = {}
    return st.session_state["bias_reads"]


def get_cached_bias(chart_id: str) -> Optional[VisualBiasResult]:
    return _bias_reads().get(chart_id)


def run_bias_read(chart: ChartUpload, instrument_hint: str) -> VisualBiasResult:
    image_bytes = get_chart_image(chart.chart_id)
    if image_bytes is None:
        result = VisualBiasResult(success=False, failure_reason="Chart image no longer available.")
    else:
        provider = get_configured_ai_provider()
        if not provider.get_capabilities().supports_vision:
            result = VisualBiasResult(
                success=False,
                failure_reason="No AI vision provider is connected — see Settings.",
            )
        else:
            result = provider.analyze_visual_bias(image_bytes, chart.timeframe_role, instrument_hint)
    _bias_reads()[chart.chart_id] = result
    return result


class TimeframeComparison(BaseModel):
    """Session-level Phase 7 output. Explicitly not a ConfluenceResult —
    no quality score exists yet (Phase 9)."""

    htf_bias: Optional[Bias] = None
    mtf_bias: Optional[Bias] = None
    ltf_bias: Optional[Bias] = None
    all_read_successfully: bool
    agreement_note: str


def compute_timeframe_comparison(session: AnalysisSession) -> Optional[TimeframeComparison]:
    """Reads cached bias results for all three of this session's charts
    and produces the agreement/conflict narrative. Returns None if any
    chart hasn't been read yet (caller should prompt the user to run
    reads first, never silently substitute a guess)."""

    if not required_charts_present(session):
        return None

    chart_ids = {"HTF": session.htf_chart_id, "MTF": session.mtf_chart_id, "LTF": session.ltf_chart_id}
    results = {role: get_cached_bias(cid) for role, cid in chart_ids.items()}
    if any(r is None for r in results.values()):
        return None

    all_ok = all(r.success for r in results.values())
    htf_b = results["HTF"].visual_bias if results["HTF"].success else None
    mtf_b = results["MTF"].visual_bias if results["MTF"].success else None
    ltf_b = results["LTF"].visual_bias if results["LTF"].success else None

    if not all_ok:
        note = "One or more timeframes couldn't be read — see individual errors above. No comparison can be drawn yet."
    elif htf_b is None or mtf_b is None or ltf_b is None:
        note = "At least one timeframe's visual impression was unclear to the AI — no confident comparison can be drawn."
    elif htf_b == mtf_b == ltf_b:
        note = f"✅ Agreement: HTF, MTF, and LTF all show a {htf_b.value} visual impression."
    elif htf_b == mtf_b and ltf_b != htf_b:
        note = (
            f"⚠️ HTF and MTF both show a {htf_b.value} impression, but LTF currently shows {ltf_b.value}. "
            "Per this project's rule, the lower timeframe should never automatically override the higher "
            "ones — this is flagged as a conflict, not resolved for you."
        )
    else:
        note = f"❓ Timeframes show a mixed picture: HTF={htf_b.value}, MTF={mtf_b.value}, LTF={ltf_b.value}."

    return TimeframeComparison(
        htf_bias=htf_b, mtf_bias=mtf_b, ltf_bias=ltf_b, all_read_successfully=all_ok, agreement_note=note
    )


# =====================================================================
# SECTION 3E: LEVEL CALCULATION ENGINE (Phase 10)
#
# Two strictly separate halves, matching the project's core rule
# ("AI explains, engine calculates"):
#   1. extract_price_levels() (Phase 10, in the AI provider above) —
#      the AI reads real numbers off the LTF chart's visible price
#      axis/labels. Nothing here asks it to calculate anything.
#   2. calculate_hypothetical_setup() below — 100% plain deterministic
#      Python math on those numbers. No AI call happens in this
#      function. If the numbers aren't there, it refuses to proceed
#      rather than filling gaps with invented prices.
# =====================================================================


def _price_reads() -> Dict[str, PriceLevelReading]:
    if "price_reads" not in st.session_state:
        st.session_state["price_reads"] = {}
    return st.session_state["price_reads"]


def get_cached_price_read(chart_id: str) -> Optional[PriceLevelReading]:
    return _price_reads().get(chart_id)


def run_price_extraction(chart: ChartUpload, instrument_hint: str) -> PriceLevelReading:
    image_bytes = get_chart_image(chart.chart_id)
    if image_bytes is None:
        result = PriceLevelReading(success=False, failure_reason="Chart image no longer available.")
    else:
        provider = get_configured_ai_provider()
        if not provider.get_capabilities().supports_vision:
            result = PriceLevelReading(success=False, failure_reason="No AI vision provider is connected — see Settings.")
        else:
            result = provider.extract_price_levels(image_bytes, instrument_hint)
    _price_reads()[chart.chart_id] = result
    return result


class SetupDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class ValidSetupResult(BaseModel):
    status: Literal["VALID_SETUP"] = "VALID_SETUP"
    session_id: str
    direction: SetupDirection
    hypothetical_entry: float
    hypothetical_stop: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    setup_quality_score: float
    setup_quality_label: SetupQualityLabel
    supporting_notes: List[str]
    invalidation_explanation: str


class NoValidSetupResult(BaseModel):
    status: Literal["NO_VALID_SETUP"] = "NO_VALID_SETUP"
    session_id: str
    reasons: List[str]


SetupCalculationResult = Union[ValidSetupResult, NoValidSetupResult]


def calculate_hypothetical_setup(session: AnalysisSession) -> SetupCalculationResult:
    """The Level Calculation Engine. Deterministic — zero AI calls in
    this function. Gates on, in order:
      1. Confluence must be computable with a real candidate direction
         (Phase 9) — no direction, no setup.
      2. No ACTIVE conflicts between timeframes (an opposite reading on
         MTF or LTF) — matches the project's explicit rule that a
         conflict must block a forced trade, not just lower a score.
      3. Real price numbers must have been read off the LTF chart
         (current price + a stop reference point in the right
         direction) — missing data means "cannot calculate," never a
         guessed number.
      4. The resulting risk must be positive and produce at least a
         1:1.2 reward — otherwise the math itself says this isn't a
         usable setup.
    Target ladder (1.5R / 3R) is a fixed, documented, consistent rule —
    not AI-chosen, not varied per chart.
    """

    reasons: List[str] = []
    conf = compute_confluence(session)
    if not conf.computable:
        return NoValidSetupResult(session_id=session.session_id, reasons=[conf.reason_if_not_computable])

    active_conflicts = [c for c in conf.conflicts if c.severity == "active"]
    if active_conflicts:
        return NoValidSetupResult(
            session_id=session.session_id,
            reasons=["A timeframe actively conflicts with the candidate direction — no forced trade:"]
            + [c.description for c in active_conflicts],
        )

    ltf_chart_id = session.ltf_chart_id
    price_read = get_cached_price_read(ltf_chart_id) if ltf_chart_id else None
    if price_read is None:
        return NoValidSetupResult(
            session_id=session.session_id,
            reasons=["LTF price levels haven't been read yet — run \"Extract price levels\" first."],
        )
    if not price_read.success or price_read.current_price is None:
        return NoValidSetupResult(
            session_id=session.session_id,
            reasons=["Exact levels cannot be calculated reliably from this screenshot.",
                      price_read.failure_reason or price_read.confidence_notes or "Current price could not be read."],
        )

    direction = conf.candidate_direction
    entry = price_read.current_price

    if direction == Bias.BULLISH:
        stop_ref = price_read.recent_swing_low
        if stop_ref is None or stop_ref >= entry:
            return NoValidSetupResult(
                session_id=session.session_id,
                reasons=["Exact levels cannot be calculated reliably from this screenshot.",
                         "No usable recent swing low below current price was read for the stop."],
            )
        stop = stop_ref * 0.999  # small buffer below the swing low
        risk = entry - stop
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 3.0
        setup_direction = SetupDirection.LONG
    elif direction == Bias.BEARISH:
        stop_ref = price_read.recent_swing_high
        if stop_ref is None or stop_ref <= entry:
            return NoValidSetupResult(
                session_id=session.session_id,
                reasons=["Exact levels cannot be calculated reliably from this screenshot.",
                         "No usable recent swing high above current price was read for the stop."],
            )
        stop = stop_ref * 1.001  # small buffer above the swing high
        risk = stop - entry
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 3.0
        setup_direction = SetupDirection.SHORT
    else:
        return NoValidSetupResult(session_id=session.session_id, reasons=["Candidate direction is neutral — no directional setup."])

    if risk <= 0:
        return NoValidSetupResult(session_id=session.session_id, reasons=["Calculated risk was zero or negative — invalid setup."])

    rr1 = abs(tp1 - entry) / risk
    if rr1 < 1.2:
        return NoValidSetupResult(
            session_id=session.session_id,
            reasons=[f"Risk/reward at TP1 ({rr1:.2f}) is below the 1:1.2 minimum — not a usable setup."],
        )

    return ValidSetupResult(
        session_id=session.session_id,
        direction=setup_direction,
        hypothetical_entry=round(entry, 3),
        hypothetical_stop=round(stop, 3),
        take_profit_1=round(tp1, 3),
        take_profit_2=round(tp2, 3),
        risk_reward_tp1=round(rr1, 2),
        risk_reward_tp2=round(abs(tp2 - entry) / risk, 2),
        setup_quality_score=conf.quality_breakdown.total,
        setup_quality_label=conf.quality_label,
        supporting_notes=conf.agreements,
        invalidation_explanation=(
            f"This hypothetical setup is invalidated if price moves past {round(stop, 3)} "
            f"(the {'swing low' if direction == Bias.BULLISH else 'swing high'} used as the stop reference), "
            "or if a fresh chart read shows a materially different picture."
        ),
    )


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
.tl-badge-bearish { color: #e2564f; border-color: #e2564f55; background: #e2564f18; }

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
        '<div class="tl-footer">TradeLens AI · Quick Scan + Multi-Timeframe · Educational tool, not financial advice</div>',
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
        + badge(f"{len(STRATEGY_REGISTRY)} strategies", "neutral")
        + badge(f"{len(EXAMPLE_INSTRUMENTS)} example instruments", "accent")
        + "</div>",
        unsafe_allow_html=True,
    )

    sessions = list_sessions()
    if sessions:
        card(
            f"{len(sessions)} active session(s)",
            "Open <b>Active Setups</b> to view or delete them. Remember: these "
            "live only in this browser tab until a database is added in a later phase.",
            icon="📈",
        )
    else:
        empty_state(
            "📭",
            "No active analyses yet",
            "Sessions you start from <b>Analyze</b> will appear here, each tracking "
            "its own HTF / MTF / LTF charts independently.",
        )
    card(
        "Quick Scan is here",
        "Upload just ONE chart on Analyze for a full AI scan — bias, entry/SL/TP, evidence "
        "score, and reasoning, no HTF/MTF/LTF required. Multi-Timeframe Scan (3 charts, "
        "independently cross-checked math) is still available as a second mode.",
        icon="🔍",
    )
    card(
        "Important",
        "TradeLens AI is an educational analysis and paper-trading tool. "
        "It does not execute real trades and does not guarantee any outcome.",
        icon="⚠️",
    )
    footer()


def render_quick_scan_result(scan: QuickScanResult, instrument_label: str, style_label: str) -> None:
    """Renders a QuickScanResult in the professional scanner-card format,
    using this app's existing card/badge components."""

    st.markdown("### 📊 AI Chart Scanner")
    st.caption(f"Instrument: {instrument_label}  ·  Mode: {style_label}")

    if not scan.success:
        st.error(scan.failure_reason or "AI analysis is temporarily unavailable.", icon="🚫")
        return

    if not scan.image_readable:
        st.warning(
            "Chart cannot be reliably analyzed"
            + (f": {scan.image_quality_note}" if scan.image_quality_note else ".")
            + " Upload a screenshot showing the candles, price scale, and enough recent price history.",
            icon="🖼️",
        )
        return

    # Shown regardless of bias — market structure/zone reads are useful
    # context even on a NO TRADE result. Everything here is the AI's own
    # visual judgment, not an independently coded geometric detector —
    # labeled as such throughout rather than presented as verified fact.
    has_concepts = any([
        scan.smc_structure_note, scan.smc_liquidity_notes, scan.order_blocks,
        scan.fair_value_gaps, scan.premium_discount_zone, scan.supply_zones,
        scan.demand_zones, scan.crt_phase,
    ])
    if has_concepts:
        with st.expander("🧠 Market Structure & Concepts (AI-identified)", expanded=(scan.bias is not None)):
            st.caption("These are the AI's own visual read — not independently verified detectors.")
            if scan.smc_structure_note or scan.smc_liquidity_notes or scan.premium_discount_zone or scan.order_blocks or scan.fair_value_gaps:
                st.markdown("**Smart Money Concepts (SMC)**")
                if scan.smc_structure_note:
                    st.write(scan.smc_structure_note)
                if scan.premium_discount_zone:
                    st.caption(f"Zone: {scan.premium_discount_zone}")
                for n in scan.smc_liquidity_notes:
                    st.write(f"• {n}")
                for ob in scan.order_blocks:
                    rng = f" ({ob.low}–{ob.high})" if ob.low is not None and ob.high is not None else ""
                    st.write(f"Order block — {ob.label}{rng}" + (f": {ob.note}" if ob.note else ""))
                for fvg in scan.fair_value_gaps:
                    rng = f" ({fvg.low}–{fvg.high})" if fvg.low is not None and fvg.high is not None else ""
                    st.write(f"FVG — {fvg.label}{rng}" + (f": {fvg.note}" if fvg.note else ""))
            if scan.supply_zones or scan.demand_zones:
                st.markdown("**Supply & Demand**")
                for z in scan.supply_zones:
                    rng = f" ({z.low}–{z.high})" if z.low is not None and z.high is not None else ""
                    st.write(f"Supply zone — {z.label}{rng}" + (f": {z.note}" if z.note else ""))
                for z in scan.demand_zones:
                    rng = f" ({z.low}–{z.high})" if z.low is not None and z.high is not None else ""
                    st.write(f"Demand zone — {z.label}{rng}" + (f": {z.note}" if z.note else ""))
            if scan.crt_phase or scan.crt_notes:
                st.markdown("**CRT (Candle Range Theory)**")
                if scan.crt_phase:
                    st.markdown(badge(scan.crt_phase, "accent"), unsafe_allow_html=True)
                if scan.crt_notes:
                    st.write(scan.crt_notes)

    if scan.bias is None:
        st.markdown("## 🚫 NO TRADE")
        st.write(scan.no_trade_reason or "No coherent, evidence-supported setup was found on this chart.")
        if scan.levels_discarded_reason and scan.levels_discarded_reason != scan.no_trade_reason:
            st.caption(f"Note: {scan.levels_discarded_reason}")
    else:
        dir_word = "LONG" if scan.bias == Bias.BULLISH else "SHORT"
        dir_tone = "bullish" if scan.bias == Bias.BULLISH else "bearish"
        st.markdown(f"## {badge(dir_word, dir_tone)}", unsafe_allow_html=True)
        if scan.market_structure:
            st.write(scan.market_structure)

        entry_txt = (
            f"{scan.entry_zone_low} – {scan.entry_zone_high}" if scan.entry_zone_low and scan.entry_zone_high
            else str(scan.entry_zone_low or scan.entry_zone_high or scan.current_price)
        )
        lines = [f"**Entry zone:** {entry_txt}", f"**Stop loss:** {scan.stop_loss}"]
        if scan.take_profit_1 is not None:
            rr = f"  (R:R 1:{scan.risk_reward_tp1})" if scan.risk_reward_tp1 else ""
            lines.append(f"**Take profit 1:** {scan.take_profit_1}{rr}")
        if scan.take_profit_2 is not None:
            rr = f"  (R:R 1:{scan.risk_reward_tp2})" if scan.risk_reward_tp2 else ""
            lines.append(f"**Take profit 2:** {scan.take_profit_2}{rr}")
        if scan.take_profit_3 is not None:
            lines.append(f"**Take profit 3:** {scan.take_profit_3}")
        st.markdown("\n\n".join(lines))

        conf_tone = {"HIGH": "bullish", "MODERATE": "accent", "LOW": "neutral", "NO_TRADE": "neutral"}
        if scan.confluence_level:
            st.markdown(badge(f"Confluence: {scan.confluence_level}", conf_tone.get(scan.confluence_level, "neutral")), unsafe_allow_html=True)
        if scan.confidence_score is not None:
            st.caption(f"Evidence score: {scan.confidence_score:.0f}/100 — NOT a win probability.")

        if scan.support_levels or scan.resistance_levels:
            st.markdown("**Key levels**")
            if scan.support_levels:
                st.write("Support: " + ", ".join(str(x) for x in scan.support_levels))
            if scan.resistance_levels:
                st.write("Resistance: " + ", ".join(str(x) for x in scan.resistance_levels))

        if scan.strategy_signals:
            st.markdown("**Strategies supporting the setup**")
            for s in scan.strategy_signals:
                st.write(f"✓ {s}")
        if scan.conflicts:
            st.markdown("**Risks / conflicts**")
            for c in scan.conflicts:
                st.write(f"⚠️ {c}")
        if scan.invalidation:
            st.caption(f"Invalidation: {scan.invalidation}")
        if scan.reasoning:
            with st.expander("AI explanation"):
                st.write(scan.reasoning)

        st.warning(
            "AI-generated technical analysis from a single chart — not independently "
            "cross-checked like Multi-Timeframe Scan, not a guarantee of any outcome, "
            "and not financial advice.",
            icon="⚠️",
        )

    if scan.patterns:
        st.caption("Patterns noted: " + ", ".join(scan.patterns))
    if scan.trend:
        st.caption(f"Visible trend: {scan.trend}")


def page_analyze() -> None:
    page_header("Analyze")
    st.markdown("**Instrument**")
    query = st.text_input(
        "Search symbol, name or market",
        placeholder="e.g. EUR/USD, NAS100, USTEC, BTC/USDT",
        label_visibility="collapsed",
    )

    resolved_instrument: Optional[Instrument] = None
    instrument_label = "Unspecified"

    if query:
        exact = instrument_resolver.resolve_exact(query)
        if exact:
            resolved_instrument = exact
            instrument_label = exact.symbol
            aliases = instrument_resolver.aliases_for(exact.instrument_id)
            alias_note = f" · also known as: {', '.join(aliases)}" if aliases else ""
            card(f"{exact.symbol} resolved", f"{exact.display_name} · {exact.asset_class.value}{alias_note}", icon="✅")
        else:
            results = instrument_resolver.search(query)
            if results:
                st.caption(f"{len(results)} possible match(es) — tap Quick Scan below to use \"{query}\" as typed, or refine your search:")
                for inst in results[:8]:
                    st.write(f"**{inst.symbol}** — {inst.display_name}")
                instrument_label = query.strip().upper()
            else:
                st.caption(f"\"{query}\" isn't in the small example list, but you can still Quick Scan with it typed as-is below.")
                instrument_label = query.strip().upper()

    st.markdown("---")
    st.markdown("**Trading style**")
    style_choice = st.radio(
        "Trading style", ["Scalping", "Day Trading", "Swing Trading", "Auto (let AI decide)"],
        horizontal=False, label_visibility="collapsed", key="quick_scan_style",
    )
    style_hint_map = {
        "Scalping": "scalping (short-term, immediate structure, momentum, and liquidity)",
        "Day Trading": "day trading (intraday structure with higher-timeframe context)",
        "Swing Trading": "swing trading (major structure, larger moves, wider stops)",
        "Auto (let AI decide)": None,
    }

    st.markdown("---")
    scan_mode = st.radio(
        "Scan mode",
        ["🔍 Quick Scan (1 chart) — default", "📊 Multi-Timeframe Scan (3 charts, more thorough)"],
        label_visibility="collapsed", key="scan_mode_choice",
    )

    if scan_mode.startswith("🔍"):
        st.caption(
            "Upload one chart screenshot. Works from any platform (TradingView, MetaTrader, "
            "broker apps, etc.). This does NOT require HTF/MTF/LTF — one chart is a complete, "
            "valid scan on its own."
        )
        uploaded = st.file_uploader("Upload chart", type=["png", "jpg", "jpeg"], key="quick_scan_upload")

        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            quality, quality_notes, width, height = analyze_chart_quality(image_bytes)
            quality_tone = {ChartQuality.GOOD: "bullish", ChartQuality.FAIR: "accent", ChartQuality.POOR: "bearish"}
            st.markdown(
                badge(f"Image quality (automatic): {quality.value}", quality_tone.get(quality, "neutral")),
                unsafe_allow_html=True,
            )
            for note in quality_notes:
                st.caption(f"⚠️ {note}")
            st.image(image_bytes, use_container_width=True)

            if st.button("🔎 Analyze Chart", type="primary"):
                provider = get_configured_ai_provider()
                if not provider.get_capabilities().supports_vision:
                    st.error("No AI vision provider is connected — see Settings to add a free Gemini API key.", icon="⚠️")
                else:
                    with st.spinner("Analyzing chart..."):
                        result = provider.quick_scan_chart(
                            image_bytes,
                            instrument_hint=instrument_label if instrument_label != "Unspecified" else None,
                            style_hint=style_hint_map[style_choice],
                        )
                    st.session_state["quick_scan_result"] = result
                    st.session_state["quick_scan_meta"] = (instrument_label, style_choice)

        if "quick_scan_result" in st.session_state:
            st.markdown("---")
            label, style = st.session_state["quick_scan_meta"]
            render_quick_scan_result(st.session_state["quick_scan_result"], label, style)

    else:
        st.caption(
            "Uses three charts (HTF/MTF/LTF) for deeper, independently cross-checked analysis "
            "with the confluence and level-calculation engines. More thorough, more setup."
        )
        if resolved_instrument is not None:
            if st.button(f"Start Multi-Timeframe session for {resolved_instrument.symbol}"):
                new_session = create_session(resolved_instrument)
                st.success(f"Session {new_session.session_id} created. Manage its charts on Active Setups.")
        else:
            with st.expander("Add instrument manually"):
                st.caption("Your manual entry always overrides automatic matching.")
                manual_class = st.selectbox("Asset class", [c.value for c in AssetClass], key="manual_asset_class")
                if st.button("Use this instrument for now"):
                    draft_instrument = Instrument(
                        instrument_id=f"manual_{_normalize(query or 'unknown')}",
                        display_name=query or "Unknown",
                        symbol=(query or "UNKNOWN").upper(),
                        asset_class=AssetClass(manual_class),
                        base_asset=(query or "UNKNOWN").upper(),
                        data_provider="manual",
                    )
                    new_session = create_session(draft_instrument)
                    st.success(f"Session {new_session.session_id} created ({manual_class}, unresolved/manual). See Active Setups.")

    footer()


def page_active_setups() -> None:
    page_header("Active Setups")
    sessions = list_sessions()
    if not sessions:
        empty_state(
            "📭",
            "No sessions yet",
            "Start one from <b>Analyze</b> by resolving or manually adding an "
            "instrument. Sessions live only in this browser tab for now — no "
            "database yet, so refreshing or closing the app clears them.",
        )
        footer()
        return

    st.caption(
        f"{len(sessions)} session(s) in this browser tab. Each is locked to one "
        "instrument — its charts can never mix with another session's."
    )
    status_tone = {
        AnalysisStatus.WAITING_FOR_CHARTS: "neutral",
        AnalysisStatus.READY: "accent",
        AnalysisStatus.ANALYZING: "accent",
        AnalysisStatus.COMPLETE: "bullish",
        AnalysisStatus.ERROR: "neutral",
    }
    role_labels = {
        TimeframeRole.HTF: "HTF — Higher Time Frame",
        TimeframeRole.MTF: "MTF — Middle Time Frame",
        TimeframeRole.LTF: "LTF — Lower Time Frame",
    }

    for s in sessions:
        chart_ids = {
            TimeframeRole.HTF: s.htf_chart_id,
            TimeframeRole.MTF: s.mtf_chart_id,
            TimeframeRole.LTF: s.ltf_chart_id,
        }
        chart_count = sum(x is not None for x in chart_ids.values())

        card(
            f"{s.instrument_symbol}  ·  #{s.session_id}",
            badge(s.status.value.replace("_", " "), status_tone[s.status])
            + f"<br><br>{chart_count}/3 charts attached · created {s.created_at}",
            icon="📈",
        )

        with st.expander(f"Manage charts — {s.instrument_symbol} #{s.session_id}"):
            quality_tone = {ChartQuality.GOOD: "bullish", ChartQuality.FAIR: "accent", ChartQuality.POOR: "bearish", ChartQuality.UNKNOWN: "neutral"}
            for role in [TimeframeRole.HTF, TimeframeRole.MTF, TimeframeRole.LTF]:
                st.markdown(f"**{role_labels[role]}**")
                existing_id = chart_ids[role]
                if existing_id:
                    chart_obj = _chart_store().get(existing_id)
                    image_bytes = get_chart_image(existing_id)
                    if image_bytes:
                        st.image(image_bytes, use_container_width=True)
                    if chart_obj:
                        dims = f"{chart_obj.width_px}×{chart_obj.height_px}px" if chart_obj.width_px else "unknown size"
                        st.markdown(
                            badge(f"Quality: {chart_obj.quality.value}", quality_tone[chart_obj.quality])
                            + f' <span style="color:#8b97a6;font-size:0.8rem;">· {dims}</span>',
                            unsafe_allow_html=True,
                        )
                        if chart_obj.quality_notes:
                            for note in chart_obj.quality_notes:
                                st.caption(f"⚠️ {note}" if chart_obj.quality != ChartQuality.GOOD else note)
                        if chart_obj.quality == ChartQuality.POOR:
                            st.warning("Please upload a clearer chart for reliable results later.", icon="⚠️")

                        ai_result_key = f"ai_result_{existing_id}"
                        if st.button(f"🔍 Read {role.value} chart with AI (beta)", key=f"airead_{s.session_id}_{role.value}"):
                            provider = get_configured_ai_provider()
                            caps = provider.get_capabilities()
                            if not caps.supports_vision:
                                st.session_state[ai_result_key] = ChartInterpretationResult(
                                    success=False,
                                    failure_reason=(
                                        "No AI vision provider is connected. Set AI_PROVIDER=gemini "
                                        "and add an API key in Secrets — see Settings for details."
                                    ),
                                )
                            else:
                                with st.spinner("Asking Gemini to read this chart..."):
                                    st.session_state[ai_result_key] = provider.interpret_chart(
                                        image_bytes, instrument_hint=s.instrument_symbol
                                    )

                        if ai_result_key in st.session_state:
                            res: ChartInterpretationResult = st.session_state[ai_result_key]
                            if res.success:
                                st.success("AI read this chart (raw observations only, not a trading conclusion):")
                                if res.detected_instrument_symbol:
                                    st.write(f"Detected instrument (unverified): **{res.detected_instrument_symbol}**")
                                if res.detected_timeframe:
                                    st.write(f"Detected timeframe (unverified): **{res.detected_timeframe}**")
                                if res.visible_candle_count:
                                    st.write(f"Estimated visible candles: **{res.visible_candle_count}**")
                                if res.raw_observations:
                                    st.markdown("**Observations:**")
                                    for obs in res.raw_observations:
                                        st.write(f"- {obs}")
                                st.caption(
                                    "This is what the AI could visibly read — nothing here is trading "
                                    "advice, and none of it feeds a strategy yet. That's a later phase."
                                )
                            else:
                                st.warning(res.failure_reason or "AI could not read this chart.")

                    if st.button(f"Remove {role.value} chart", key=f"remove_{s.session_id}_{role.value}"):
                        clear_chart_role(s, role)
                        st.rerun()
                else:
                    uploaded = st.file_uploader(
                        f"Upload {role.value} chart",
                        type=["png", "jpg", "jpeg"],
                        key=f"upload_{s.session_id}_{role.value}",
                        label_visibility="collapsed",
                    )
                    if uploaded is not None:
                        attach_chart(s, role, uploaded)
                        st.rerun()

            if chart_count == 3:
                st.success("All 3 charts attached.")
            else:
                missing = [role.value for role, cid in chart_ids.items() if cid is None]
                st.caption(f"Waiting for: {', '.join(missing)}")
            st.caption(
                "Quality checks here (resolution, cropping, blur estimate, duplicates) "
                "are automatic image analysis only — not AI, and they don't verify the "
                "chart's actual content (instrument, timeframe, candles). That's the AI "
                "vision buttons above."
            )

        if chart_count == 3:
            with st.expander(f"Compare timeframes (AI, beta) — {s.instrument_symbol} #{s.session_id}"):
                st.caption(
                    "Reads each of HTF/MTF/LTF and compares their visual price-shape "
                    "impressions. This is NOT a trading signal, evidence, or a quality "
                    "score — those require real strategy modules (a later phase)."
                )
                if st.button("📊 Run comparison", key=f"compare_{s.session_id}"):
                    with st.spinner("Reading HTF, MTF, and LTF..."):
                        for role, cid in chart_ids.items():
                            chart_obj = _chart_store().get(cid)
                            if chart_obj is not None:
                                run_bias_read(chart_obj, s.instrument_symbol)
                    st.rerun()

                comparison = compute_timeframe_comparison(s)
                bias_tone = {Bias.BULLISH: "bullish", Bias.BEARISH: "bearish", Bias.NEUTRAL: "neutral"}
                if comparison is None:
                    for role, cid in chart_ids.items():
                        cached = get_cached_bias(cid) if cid else None
                        if cached is None:
                            st.caption(f"{role.value}: not read yet.")
                        elif not cached.success:
                            st.warning(f"{role.value}: {cached.failure_reason}")
                        else:
                            label = cached.visual_bias.value if cached.visual_bias else "unclear"
                            st.markdown(
                                f"{role.value}: " + badge(label, bias_tone.get(cached.visual_bias, "neutral")),
                                unsafe_allow_html=True,
                            )
                            if cached.reasoning:
                                st.caption(cached.reasoning)
                else:
                    for role, bias_val in [("HTF", comparison.htf_bias), ("MTF", comparison.mtf_bias), ("LTF", comparison.ltf_bias)]:
                        label = bias_val.value if bias_val else "unclear"
                        st.markdown(f"{role}: " + badge(label, bias_tone.get(bias_val, "neutral")), unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(comparison.agreement_note)

            with st.expander(f"Strategy Engine (beta) — {s.instrument_symbol} #{s.session_id}"):
                st.caption(
                    f"Runs {len(STRATEGY_REGISTRY)} registered module(s) against the SAME cached "
                    "reads above — no extra AI calls. Each result includes whether there was "
                    "enough evidence, per the project's rule that missing evidence must be shown, "
                    "never hidden."
                )
                strategy_results = run_strategy_engine(s)
                layer_order = [
                    EvidenceLayer.MARKET_CONTEXT, EvidenceLayer.MARKET_LOCATION,
                    EvidenceLayer.LIQUIDITY_STRUCTURE, EvidenceLayer.CONFIRMATION, EvidenceLayer.EXECUTION,
                ]
                by_layer = {layer: [r for r in strategy_results if r.layer == layer] for layer in layer_order}
                for layer in layer_order:
                    layer_results = by_layer[layer]
                    if not layer_results:
                        continue
                    st.markdown(f"**{layer.value.replace('_', ' ').title()}**")
                    for r in layer_results:
                        if r.insufficient_data:
                            st.warning(f"{r.timeframe_role.value} · {r.strategy_name}: {'; '.join(r.notes)}", icon="⚠️")
                        else:
                            st.markdown(
                                f"{r.timeframe_role.value} · {r.strategy_name}: "
                                + badge(r.bias.value, bias_tone.get(r.bias, "neutral")),
                                unsafe_allow_html=True,
                            )
                            for ev in r.evidence:
                                st.caption(f"↳ {ev}")
                st.caption(
                    "See \"Confluence & Setup Quality\" below for how this evidence "
                    "combines into one transparent score."
                )

            with st.expander(f"Confluence & Setup Quality (beta) — {s.instrument_symbol} #{s.session_id}"):
                st.caption(
                    "Combines the Strategy Engine's evidence above into ONE transparent "
                    "setup-quality score — reusing cached data, no new AI calls. This is a "
                    "quality score, NEVER a win probability."
                )
                conf = compute_confluence(s)
                if not conf.computable:
                    st.warning(conf.reason_if_not_computable, icon="⚠️")
                else:
                    b = conf.quality_breakdown
                    st.markdown(
                        f"**Candidate direction:** "
                        + badge(conf.candidate_direction.value, bias_tone.get(conf.candidate_direction, "neutral")),
                        unsafe_allow_html=True,
                    )
                    label_tone = {
                        SetupQualityLabel.WEAK_NO_SETUP: "neutral", SetupQualityLabel.LOW_QUALITY: "neutral",
                        SetupQualityLabel.MODERATE: "accent", SetupQualityLabel.STRONG: "bullish",
                        SetupQualityLabel.VERY_STRONG: "bullish",
                    }
                    st.markdown(
                        f"**Setup Quality: {b.total:.0f}/100** "
                        + badge(conf.quality_label.value.replace("_", " "), label_tone[conf.quality_label]),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"- HTF alignment: {b.htf_alignment:.0f}/20\n"
                        f"- MTF alignment: {b.mtf_alignment:.0f}/20\n"
                        f"- LTF confirmation: {b.ltf_confirmation:.0f}/20\n"
                        f"- Liquidity: {b.liquidity:.0f}/10\n"
                        f"- Structure: {b.structure:.0f}/10\n"
                        f"- Location: {b.location:.0f}/10\n"
                        f"- Risk/Reward: {b.risk_reward:.0f}/10"
                    )
                    if conf.agreements:
                        st.markdown("**Agreements:**")
                        for a in conf.agreements:
                            st.write(f"✓ {a}")
                    if conf.conflicts:
                        st.markdown("**Conflicts / gaps (never hidden):**")
                        for c in conf.conflicts:
                            st.write(f"⚠️ {c.description}")
                    with st.expander("Why the unscored dimensions show 0"):
                        for note in conf.unscored_dimension_notes:
                            st.caption(note)
                    st.caption(
                        "This score currently maxes out around 60/100 (LOW QUALITY) even with "
                        "full agreement — that ceiling is intentional. It rises only once more "
                        "strategy modules and real price levels exist (later phases), never by "
                        "estimating what isn't there."
                    )

            with st.expander(f"Hypothetical Setup (beta) — {s.instrument_symbol} #{s.session_id}"):
                st.caption(
                    "Reads REAL numbers off the LTF chart's price axis (not estimated), then "
                    "calculates entry/stop/targets with plain deterministic math — never AI-guessed "
                    "prices. This is educational and hypothetical only, not trading advice."
                )
                ltf_chart_obj = _chart_store().get(s.ltf_chart_id) if s.ltf_chart_id else None
                if ltf_chart_obj is not None:
                    if st.button("📐 Extract price levels from LTF chart", key=f"extract_{s.session_id}"):
                        with st.spinner("Reading price labels off the LTF chart..."):
                            run_price_extraction(ltf_chart_obj, s.instrument_symbol)
                        st.rerun()

                    price_read = get_cached_price_read(s.ltf_chart_id)
                    if price_read is not None:
                        if price_read.success:
                            st.write(
                                f"Current price: **{price_read.current_price}**  ·  "
                                f"Swing high: **{price_read.recent_swing_high}**  ·  "
                                f"Swing low: **{price_read.recent_swing_low}**"
                            )
                            if price_read.confidence_notes:
                                st.caption(price_read.confidence_notes)
                        else:
                            st.warning(price_read.failure_reason, icon="⚠️")

                calc = calculate_hypothetical_setup(s)
                st.markdown("---")
                if calc.status == "NO_VALID_SETUP":
                    st.markdown("### 🚫 NO VALID SETUP")
                    for r in calc.reasons:
                        st.write(f"- {r}")
                else:
                    dir_tone = "bullish" if calc.direction.value == "long" else "bearish"
                    st.markdown(f"### Hypothetical {calc.direction.value.upper()} setup")
                    st.markdown(badge(f"Setup Quality: {calc.setup_quality_score:.0f}/100 ({calc.setup_quality_label.value.replace('_',' ')})", dir_tone), unsafe_allow_html=True)
                    st.markdown(
                        f"**Hypothetical Entry:** {calc.hypothetical_entry}\n\n"
                        f"**Hypothetical Stop / Invalidation:** {calc.hypothetical_stop}\n\n"
                        f"**TP1:** {calc.take_profit_1}  (R:R 1:{calc.risk_reward_tp1})\n\n"
                        f"**TP2:** {calc.take_profit_2}  (R:R 1:{calc.risk_reward_tp2})"
                    )
                    if calc.supporting_notes:
                        st.markdown("**Supporting evidence:**")
                        for n in calc.supporting_notes:
                            st.write(f"✓ {n}")
                    st.caption(f"Invalidation: {calc.invalidation_explanation}")
                    st.warning(
                        "This is a hypothetical, educational calculation from limited automated "
                        "evidence — NOT financial advice, and not a guarantee of any outcome.",
                        icon="⚠️",
                    )

        if st.button("Delete session", key=f"delete_{s.session_id}"):
            delete_session(s.session_id)
            st.rerun()
    footer()


def page_strategy_lab() -> None:
    page_header("Strategy Lab")
    st.markdown(
        '<div class="tl-status-row">' + badge(f"{len(STRATEGY_REGISTRY)} registered", "accent") + "</div>",
        unsafe_allow_html=True,
    )
    for module in STRATEGY_REGISTRY:
        layer_list = ", ".join(l.value.replace("_", " ") for l in module.layers_covered)
        card(module.name, f"{module.description}<br><br>Layers covered: {layer_list}", icon="🧠")
    empty_state(
        "🧪",
        "Only one real module so far — on purpose",
        "Named technical strategies (BOS/CHoCH, FVG, order blocks, RSI, etc.) need real "
        "structured price/indicator data this app doesn't have yet. Adding them now would "
        "mean the AI inventing structure that isn't verifiably there. They'll get added one "
        "at a time, only once there's real data to back each one.",
    )
    st.caption(
        "Run this module from a session's \"Strategy Engine\" section on Active Setups "
        "(requires all 3 charts read via Compare Timeframes first)."
    )
    card(
        "Backtesting",
        "Comes later, once more strategy modules and historical data exist. "
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
        "To connect Google Gemini's free tier: get a free API key at "
        "aistudio.google.com/apikey, then in Streamlit Cloud go to your app's "
        "⋮ menu → Settings → Secrets and add:<br>"
        '<code>AI_PROVIDER = "gemini"</code><br>'
        '<code>AI_PROVIDER_API_KEY = "your-key-here"</code>',
        icon="🤖",
    )
    warning = ai_setup_warning()
    if warning:
        st.warning(warning, icon="⚠️")
    if caps.supports_vision:
        st.caption(
            "Free-tier honesty notes: Gemini's free tier is rate-limited "
            "(roughly 10 requests/minute, ~1,500/day) and Google may use "
            "free-tier prompts/images to improve their products — avoid "
            "uploading anything sensitive. It reports only what it can see, "
            "never a trading conclusion."
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
