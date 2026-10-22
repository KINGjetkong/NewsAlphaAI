"""
backend/streaming/cross_modal_linker.py
Temporal cross-modal event linking, entity association, and composite sentiment velocity.
Associates concurrent alternative data events across news, social, trends, and order flow.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from backend.streaming.event_schema import SourceType, UnifiedEvent


class CrossModalCluster(BaseModel):
    """A cluster of temporally linked events across distinct modalities."""
    cluster_id: str
    ticker: str
    start_time: datetime
    end_time: datetime
    sources: List[str]
    event_ids: List[str]
    headline_summary: str
    average_sentiment: float
    max_urgency: float
    confluence_score: float = Field(
        description="Score between 0.0 and 1.0 indicating degree of cross-modal alignment"
    )
    is_sentiment_shock: bool = False


class SentimentVelocityMetrics(BaseModel):
    """Quantifies real-time sentiment velocity and momentum across feeds."""
    ticker: str
    timestamp: datetime
    event_count_15m: int
    event_count_1h: int
    composite_sentiment_15m: float
    composite_sentiment_1h: float
    sentiment_velocity: float = Field(
        description="d(Sentiment)/dt normalized rate of change between 15m and 1h"
    )
    source_breakdown: Dict[str, float]
    shock_detected: bool = False
    dominant_modality: str = "none"


class CrossModalLinker:
    """
    Maintains a rolling temporal window of multi-source events and computes
    entity linking, multi-modal clusters, and sentiment velocity.
    """

    # Weights representing information fidelity and market impact
    SOURCE_WEIGHTS: Dict[SourceType, float] = {
        SourceType.SEC_FILING: 1.00,
        SourceType.OPTIONS_FLOW: 0.95,
        SourceType.NEWS: 0.85,
        SourceType.POLITICAL: 0.80,
        SourceType.GOOGLE_TRENDS: 0.70,
        SourceType.SANTIMENT: 0.65,
        SourceType.STOCKTWITS: 0.50,
        SourceType.REDDIT: 0.45,
        SourceType.X_TWITTER: 0.45,
        SourceType.CUSTOM: 0.40,
    }

    def __init__(self, window_hours: float = 4.0) -> None:
        self.window_duration = timedelta(hours=window_hours)
        # ticker -> deque of UnifiedEvent
        self._event_windows: Dict[str, deque[UnifiedEvent]] = defaultdict(deque)

    def ingest_event(self, event: UnifiedEvent) -> None:
        """Add an event to the rolling window and evict expired records."""
        ticker = event.ticker.upper()
        window = self._event_windows[ticker]
        window.append(event)
        self._prune_window(ticker)

    def ingest_events(self, events: List[UnifiedEvent]) -> None:
        """Ingest a batch of events."""
        for e in events:
            self.ingest_event(e)

    def _prune_window(self, ticker: str) -> None:
        """Remove events older than window_duration."""
        window = self._event_windows[ticker]
        cutoff = datetime.now(timezone.utc) - self.window_duration
        while window and window[0].timestamp < cutoff:
            window.popleft()

    def get_recent_events(self, ticker: str, minutes: float = 60.0) -> List[UnifiedEvent]:
        """Retrieve events within the last N minutes for a ticker."""
        self._prune_window(ticker.upper())
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [
            e for e in self._event_windows[ticker.upper()]
            if e.timestamp >= cutoff
        ]

    def compute_sentiment_velocity(self, ticker: str) -> SentimentVelocityMetrics:
        """
        Calculate composite weighted sentiment score and velocity across
        short-term (15m) and medium-term (1h) windows.
        """
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)
        t_15m = now - timedelta(minutes=15)

        events_1h = self.get_recent_events(ticker, minutes=60.0)
        events_15m = [e for e in events_1h if e.timestamp >= t_15m]

        def _calc_weighted_sentiment(ev_list: List[UnifiedEvent]) -> tuple[float, Dict[str, float]]:
            if not ev_list:
                return 0.0, {}
            total_weight = 0.0
            weighted_sum = 0.0
            source_sums: Dict[str, float] = defaultdict(float)
            source_weights: Dict[str, float] = defaultdict(float)

            for e in ev_list:
                fidelity = self.SOURCE_WEIGHTS.get(e.source_type, 0.5)
                w = e.urgency_weight * fidelity * max(0.1, e.impact_score)
                total_weight += w
                weighted_sum += e.sentiment_score * w
                source_sums[e.source_type.value] += e.sentiment_score * w
                source_weights[e.source_type.value] += w

            comp = weighted_sum / total_weight if total_weight > 0 else 0.0
            breakdown = {
                src: round(source_sums[src] / source_weights[src], 3)
                for src in source_sums if source_weights[src] > 0
            }
            return comp, breakdown

        comp_15m, breakdown = _calc_weighted_sentiment(events_15m)
        comp_1h, _ = _calc_weighted_sentiment(events_1h)

        # Velocity = difference between fast momentum and baseline
        velocity = comp_15m - comp_1h

        # Shock detection: high absolute velocity or massive volume of urgent items
        shock = abs(velocity) > 0.35 and len(events_15m) >= 2

        # Dominant modality
        counts: Dict[str, int] = defaultdict(int)
        for e in events_15m:
            counts[e.source_type.value] += 1
        dominant = max(counts, key=counts.get) if counts else "none"

        return SentimentVelocityMetrics(
            ticker=ticker,
            timestamp=now,
            event_count_15m=len(events_15m),
            event_count_1h=len(events_1h),
            composite_sentiment_15m=round(comp_15m, 3),
            composite_sentiment_1h=round(comp_1h, 3),
            sentiment_velocity=round(velocity, 3),
            source_breakdown=breakdown,
            shock_detected=shock,
            dominant_modality=dominant,
        )

    def find_cross_modal_clusters(
        self,
        ticker: str,
        proximity_minutes: float = 30.0,
        min_distinct_sources: int = 2,
    ) -> List[CrossModalCluster]:
        """
        Identify clusters of events occurring in close temporal proximity
        spanning multiple different data sources (cross-modal confluence).
        """
        ticker = ticker.upper()
        events = sorted(
            self.get_recent_events(ticker, minutes=120.0),
            key=lambda e: e.timestamp,
        )
        if len(events) < min_distinct_sources:
            return []

        clusters: List[CrossModalCluster] = []
        visited = set()

        for i, ev_a in enumerate(events):
            if ev_a.event_id in visited:
                continue

            current_group: List[UnifiedEvent] = [ev_a]
            window_end = ev_a.timestamp + timedelta(minutes=proximity_minutes)

            for j in range(i + 1, len(events)):
                ev_b = events[j]
                if ev_b.timestamp <= window_end:
                    current_group.append(ev_b)
                else:
                    break

            distinct_sources = set(e.source_type.value for e in current_group)
            if len(distinct_sources) >= min_distinct_sources:
                for e in current_group:
                    visited.add(e.event_id)

                avg_sent = sum(e.sentiment_score for e in current_group) / len(current_group)
                max_urg = max(e.urgency_weight for e in current_group)

                # Confluence score measures whether sources agree on sentiment direction
                signs = [1 if e.sentiment_score > 0.1 else (-1 if e.sentiment_score < -0.1 else 0)
                         for e in current_group]
                non_neutral = [s for s in signs if s != 0]
                alignment = (
                    abs(sum(non_neutral)) / len(non_neutral)
                    if non_neutral else 0.5
                )
                source_factor = min(1.0, len(distinct_sources) / 4.0)
                confluence = round(alignment * 0.6 + source_factor * 0.4, 3)

                is_shock = confluence > 0.7 and max_urg > 0.8 and abs(avg_sent) > 0.4

                clusters.append(CrossModalCluster(
                    cluster_id=f"cluster_{ticker}_{ev_a.timestamp.strftime('%H%M%S')}",
                    ticker=ticker,
                    start_time=current_group[0].timestamp,
                    end_time=current_group[-1].timestamp,
                    sources=sorted(list(distinct_sources)),
                    event_ids=[e.event_id for e in current_group],
                    headline_summary=current_group[0].headline,
                    average_sentiment=round(avg_sent, 3),
                    max_urgency=round(max_urg, 3),
                    confluence_score=confluence,
                    is_sentiment_shock=is_shock,
                ))

        return clusters


_global_linker: Optional[CrossModalLinker] = None


def get_cross_modal_linker() -> CrossModalLinker:
    """Retrieve singleton CrossModalLinker instance."""
    global _global_linker
    if _global_linker is None:
        _global_linker = CrossModalLinker(window_hours=6.0)
    return _global_linker
