"""
TradeLens AI — Phase 6 (single-file edition)

An educational, mobile-first trading chart analysis and paper-trading
platform. This does NOT give financial advice, does NOT execute real
trades, and NEVER guarantees any outcome.

Phase 0 built the foundation. Phase 1 polished the mobile UI. Phase 2
added real instrument resolution. Phase 3 added real session
management. Phase 4 added real three-chart upload. Phase 5 added
deterministic (non-AI) chart quality checking. Phase 6 (this version)
adds real AI vision: Google Gemini's free tier (chosen over Groq's
free-tier vision, which is preview-only and less reliable) can now
actually look at an uploaded chart and report what it visibly sees —
never a trading conclusion, only raw observations. Requires the user to
add their own free Gemini API key in Streamlit Cloud Secrets; without
one, this honestly reports "not available" rather than faking a result.
The strategy engine and confluence scoring are still not implemented —
every placeholder below says so honestly instead of pretending to work.

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
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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


class AIProvider(ABC):
    @abstractmethod
    def get_capabilities(self) -> AIProviderCapabilities: ...

    @abstractmethod
    def interpret_chart(
        self, image_bytes: bytes, instrument_hint: Optional[str] = None
    ) -> ChartInterpretationResult: ...

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

    MODEL = "gemini-flash-latest"  # Google's auto-updating Flash alias — avoids
                                     # this code breaking every time a dated
                                     # model version is deprecated.

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_capabilities(self) -> AIProviderCapabilities:
        return AIProviderCapabilities(
            supports_vision=True,
            supports_text=True,
            provider_name="Google Gemini (free tier)",
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

        try:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": self._mime_type_for(image_bytes), "data": b64}},
                    ]
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 700},
            }
            resp = requests.post(url, params={"key": self._api_key}, json=payload, timeout=30)
        except requests.exceptions.Timeout:
            return ChartInterpretationResult(success=False, failure_reason="Gemini request timed out after 30 seconds.")
        except requests.exceptions.RequestException as e:
            return ChartInterpretationResult(success=False, failure_reason=f"Network error calling Gemini: {e}")

        if resp.status_code == 429:
            return ChartInterpretationResult(
                success=False,
                failure_reason="Gemini's free-tier rate limit was hit. Wait a minute and try again.",
            )
        if resp.status_code == 400:
            return ChartInterpretationResult(
                success=False,
                failure_reason="Gemini rejected this request (often an invalid API key). Check Settings/Secrets.",
            )
        if resp.status_code != 200:
            return ChartInterpretationResult(
                success=False, failure_reason=f"Gemini API returned an unexpected status ({resp.status_code})."
            )

        try:
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return ChartInterpretationResult(success=False, failure_reason="Gemini returned no result for this image.")
            text = candidates[0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            return ChartInterpretationResult(
                success=False, failure_reason=f"Gemini's response wasn't in the expected format ({e})."
            )

        return ChartInterpretationResult(
            success=True,
            detected_instrument_symbol=parsed.get("detected_instrument_symbol"),
            detected_timeframe=parsed.get("detected_timeframe"),
            visible_candle_count=parsed.get("visible_candle_count_estimate"),
            raw_observations=list(parsed.get("observations") or []),
        )

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
# SECTION 3: STRATEGY REGISTRY
# (equivalent to tradelens/engines/strategy_registry.py)
# Ships EMPTY on purpose — implementing strategies before the AI vision
# layer exists would mean fabricating analysis logic.
# =====================================================================

REGISTERED_STRATEGIES: List[str] = []


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


def get_chart_image(chart_id: str) -> Optional[bytes]:
    return _chart_images().get(chart_id)


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
        '<div class="tl-footer">TradeLens AI · Phase 6 · Educational tool, not financial advice</div>',
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
        "Phase 6 status",
        "AI vision is now real (Google Gemini free tier) — from Active Setups, "
        "you can ask the AI to read an uploaded chart and see raw observations. "
        "It never gives a trading conclusion, and the strategy engine that would "
        "turn observations into evidence is still not implemented.",
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
            if st.button(f"Start analysis session for {exact.symbol}"):
                new_session = create_session(exact)
                st.success(
                    f"Session {new_session.session_id} created for {exact.symbol}. "
                    "See it on Active Setups."
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
                        draft_instrument = Instrument(
                            instrument_id=f"manual_{_normalize(query)}",
                            display_name=query,
                            symbol=query.upper(),
                            asset_class=AssetClass(manual_class),
                            base_asset=query.upper(),
                            data_provider="manual",
                        )
                        new_session = create_session(draft_instrument)
                        st.success(
                            f"Session {new_session.session_id} created for \"{query}\" "
                            f"({manual_class}, unresolved/manual). See it on Active Setups."
                        )

    st.caption(
        "Searching a small example list across forex, indices, crypto, "
        "commodities, and stocks — not a live provider yet."
    )
    st.markdown("&nbsp;", unsafe_allow_html=True)
    card("Mode", "Scalping / Day Trading / Swing / Analyze All — coming in a later phase.", icon="⏱️")
    card("Charts", "HTF / MTF / LTF upload and quality validation — coming in a later phase.", icon="🖼️")
    empty_state("🚧", "Charts not runnable yet", "Starting a session works. Uploading charts and analyzing them does not, yet.")
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
                st.success("All 3 charts attached. Analysis itself is a later phase.")
            else:
                missing = [role.value for role, cid in chart_ids.items() if cid is None]
                st.caption(f"Waiting for: {', '.join(missing)}")
            st.caption(
                "Quality checks here (resolution, cropping, blur estimate, duplicates) "
                "are automatic image analysis only — not AI, and they don't verify the "
                "chart's actual content (instrument, timeframe, candles). That's Phase 6."
            )

        if st.button("Delete session", key=f"delete_{s.session_id}"):
            delete_session(s.session_id)
            st.rerun()
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
