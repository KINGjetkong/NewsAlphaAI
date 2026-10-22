"""
backend/streaming/event_schema.py
Standardized unified event model for multi-source data ingestion,
sentiment scoring, market ticks, and Deephaven As-Of joined records.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    NEWS = "news"
    SEC_FILING = "sec_filing"
    PRESS_RELEASE = "press_release"
    TWITTER = "twitter"
    SANTIMENT = "santiment"
    STOCKTWITS = "stocktwits"
    REDDIT = "reddit"
    GOOGLE_TRENDS = "google_trends"
    UNUSUAL_WHALES = "unusual_whales"
    POLITICAL = "political"
    MARKET_TICK = "market_tick"


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class UnifiedEvent(BaseModel):
    """
    Normalized data point across all asynchronous news, social,
    order-flow, and momentum feeds.
    """
    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str
    source_type: SourceType
    headline: str = ""
    text: str = ""
    author: Optional[str] = None
    url: Optional[str] = None
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sentiment_probs: Dict[str, float] = Field(
        default_factory=lambda: {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    )
    impact_score: float = 0.0  # 0.0 to 1.0 (confidence or magnitude)
    urgency_weight: float = 1.0  # 0.0 to 1.0 (source authority & speed)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "source_type": self.source_type.value,
            "headline": self.headline,
            "text": self.text,
            "sentiment_label": self.sentiment_label.value,
            "sentiment_score": self.sentiment_score,
            "sentiment_probs": self.sentiment_probs,
            "impact_score": self.impact_score,
            "urgency_weight": self.urgency_weight,
            "metadata": self.metadata,
        }


class MarketTick(BaseModel):
    """
    Real-time high-resolution trade print for tick-by-tick synchronization.
    """
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str
    price: float
    size: float = 1.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "price": self.price,
            "size": self.size,
            "bid": self.bid if self.bid is not None else self.price,
            "ask": self.ask if self.ask is not None else self.price,
            "volume": self.volume if self.volume is not None else self.size,
            "condition": self.condition or "@",
        }


class AsOfJoinedTick(BaseModel):
    """
    Point-in-time time-aligned record combining a market tick with the
    most recent prior sentiment, flow, and momentum signals.
    """
    timestamp: datetime
    ticker: str
    price: float
    size: float
    bid: float
    ask: float
    volume: float
    last_event_id: Optional[str] = None
    last_event_source: Optional[str] = None
    last_headline: Optional[str] = None
    last_sentiment_score: float = 0.0
    last_sentiment_label: str = "NEUTRAL"
    last_impact_score: float = 0.0
    sentiment_velocity: float = 0.0  # Delta sentiment over time window
    flow_sentiment_score: float = 0.0  # Unusual Whales sentiment
    search_momentum_score: float = 0.0  # Google Trends momentum
    price_delta_1tick: float = 0.0
    price_delta_10ticks: float = 0.0
    price_delta_60s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "price": self.price,
            "size": self.size,
            "bid": self.bid,
            "ask": self.ask,
            "volume": self.volume,
            "last_event_id": self.last_event_id,
            "last_event_source": self.last_event_source,
            "last_headline": self.last_headline,
            "last_sentiment_score": self.last_sentiment_score,
            "last_sentiment_label": self.last_sentiment_label,
            "last_impact_score": self.last_impact_score,
            "sentiment_velocity": self.sentiment_velocity,
            "flow_sentiment_score": self.flow_sentiment_score,
            "search_momentum_score": self.search_momentum_score,
            "price_delta_1tick": self.price_delta_1tick,
            "price_delta_10ticks": self.price_delta_10ticks,
            "price_delta_60s": self.price_delta_60s,
        }
