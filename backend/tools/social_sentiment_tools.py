"""
backend/tools/social_sentiment_tools.py
Asynchronous ingestion for Social Sentiment:
- X (Twitter) Cashtag Stream
- Santiment GraphQL On-Chain/Social Metrics
- StockTwits Real-time Stream
- Reddit Financial Subreddit Stream
Scores content through FinBERT to generate normalized UnifiedEvents.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import List
import httpx
from langchain_core.tools import tool
from loguru import logger

from backend.core.config import settings
from backend.streaming.circuit_breaker import AdaptiveRateLimiter, CircuitBreaker
from backend.streaming.event_schema import SentimentLabel, SourceType, UnifiedEvent
from backend.tools.finbert_tools import FinBERTSentimentAnalyzer

_social_limiter = AdaptiveRateLimiter(rate_per_second=2.0, burst=5)
_twitter_breaker = CircuitBreaker("TwitterX", failure_threshold=3, recovery_timeout=45.0)
_santiment_breaker = CircuitBreaker("Santiment", failure_threshold=3, recovery_timeout=45.0)
_stocktwits_breaker = CircuitBreaker("StockTwits", failure_threshold=3, recovery_timeout=30.0)
_reddit_breaker = CircuitBreaker("Reddit", failure_threshold=3, recovery_timeout=30.0)


class SocialSentimentIngestionEngine:
    """
    Coordinates asynchronous streaming and polling across X, Santiment,
    StockTwits, and Reddit.
    """
    def __init__(self):
        self.finbert = FinBERTSentimentAnalyzer.get_instance()

    # ─── StockTwits Ingestion ────────────────────────────────────────────────

    async def fetch_stocktwits(self, ticker: str, limit: int = 15) -> List[UnifiedEvent]:
        """Fetch real-time trader messages from StockTwits public symbol stream."""
        if not settings.stocktwits_enabled:
            return []

        async def _fetch() -> List[UnifiedEvent]:
            await _social_limiter.acquire()
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url)
                if resp.status_code == 429:
                    _social_limiter.backoff()
                    return []
                resp.raise_for_status()
                data = resp.json()
                messages = data.get("messages", [])[:limit]
                events: List[UnifiedEvent] = []

                for m in messages:
                    mid = str(m.get("id", ""))
                    body = m.get("body", "")
                    created_at_str = m.get("created_at")
                    try:
                        dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    except Exception:
                        dt = datetime.now(timezone.utc)

                    st_sentiment = m.get("entities", {}).get("sentiment", {})
                    basic_label = st_sentiment.get("basic") if st_sentiment else None

                    scored = await self.finbert.score_text(body)
                    score = scored["score"]
                    if basic_label == "Bullish":
                        score = max(score, 0.4)
                    elif basic_label == "Bearish":
                        score = min(score, -0.4)

                    label = (
                        SentimentLabel.BULLISH if score > 0.15
                        else SentimentLabel.BEARISH if score < -0.15
                        else SentimentLabel.NEUTRAL
                    )

                    events.append(UnifiedEvent(
                        event_id=f"st_{mid}",
                        timestamp=dt,
                        ticker=ticker.upper(),
                        source_type=SourceType.STOCKTWITS,
                        headline=f"StockTwits @{m.get('user', {}).get('username', 'trader')}",
                        text=body,
                        author=m.get("user", {}).get("username"),
                        sentiment_label=label,
                        sentiment_score=score,
                        sentiment_probs=scored["probs"],
                        impact_score=scored["impact_score"],
                        urgency_weight=0.7,
                        metadata={"likes": m.get("likes", {}).get("total", 0)},
                    ))
                return events

        res = await _stocktwits_breaker.call(_fetch, fallback=[])
        return res or []

    # ─── Reddit Ingestion ────────────────────────────────────────────────────

    async def fetch_reddit(self, ticker: str, limit: int = 10) -> List[UnifiedEvent]:
        """Fetch discussions from financial subreddits (r/wallstreetbets, r/stocks)."""
        async def _fetch() -> List[UnifiedEvent]:
            await _social_limiter.acquire()
            headers = {"User-Agent": "NewsAlphaAI/1.0 (financial-nlp-research)"}
            subreddits = "stocks+wallstreetbets"
            url = (
                f"https://www.reddit.com/r/{subreddits}/search.json"
                f"?q={ticker}&sort=new&limit={limit}"
            )
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                resp = await client.get(url)
                if resp.status_code in (429, 403):
                    _social_limiter.backoff()
                    return []
                resp.raise_for_status()
                data = resp.json()
                children = data.get("data", {}).get("children", [])
                events: List[UnifiedEvent] = []

                for item in children:
                    d = item.get("data", {})
                    post_id = d.get("id", "")
                    title = d.get("title", "")
                    selftext = d.get("selftext", "")[:300]
                    created_utc = d.get("created_utc", datetime.now(timezone.utc).timestamp())
                    dt = datetime.fromtimestamp(created_utc, timezone.utc)

                    scored = await self.finbert.score_text(f"{title}. {selftext}")
                    score = scored["score"]
                    label = (
                        SentimentLabel.BULLISH if score > 0.15
                        else SentimentLabel.BEARISH if score < -0.15
                        else SentimentLabel.NEUTRAL
                    )

                    events.append(UnifiedEvent(
                        event_id=f"rd_{post_id}",
                        timestamp=dt,
                        ticker=ticker.upper(),
                        source_type=SourceType.REDDIT,
                        headline=f"Reddit r/{d.get('subreddit')}: {title[:80]}",
                        text=selftext,
                        author=d.get("author"),
                        sentiment_label=label,
                        sentiment_score=score,
                        sentiment_probs=scored["probs"],
                        impact_score=scored["impact_score"],
                        urgency_weight=0.6,
                        metadata={
                            "ups": d.get("ups", 0),
                            "num_comments": d.get("num_comments", 0),
                        },
                    ))
                return events

        res = await _reddit_breaker.call(_fetch, fallback=[])
        return res or []

    # ─── Santiment Ingestion ─────────────────────────────────────────────────

    async def fetch_santiment(self, ticker: str) -> List[UnifiedEvent]:
        """Fetch social volume & weighted sentiment via Santiment GraphQL API."""
        if not settings.santiment_api_key:
            return []

        async def _fetch() -> List[UnifiedEvent]:
            await _social_limiter.acquire()
            url = "https://api.santiment.net/graphql"
            headers = {"Authorization": f"Apikey {settings.santiment_api_key}"}
            # Slug mapping for equities / digital assets
            query = """
            query ($slug: String!) {
              getMetric(metric: "sentiment_volume_consumed") {
                timeseriesData(slug: $slug, from: "utc_now-1d", to: "utc_now", interval: "1h") {
                  datetime
                  value
                }
              }
            }
            """
            slug = ticker.lower()
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                resp = await client.post(url, json={"query": query, "variables": {"slug": slug}})
                resp.raise_for_status()
                data = resp.json()
                ts_data = data.get("data", {}).get("getMetric", {}).get("timeseriesData", [])
                if not ts_data:
                    return []

                last_pt = ts_data[-1]
                val = float(last_pt.get("value", 0.0))
                score = max(-1.0, min(1.0, (val - 50.0) / 50.0))
                hash_input = f"santiment:{ticker}:{last_pt.get('datetime')}".encode()
                eid = hashlib.md5(hash_input).hexdigest()[:12]

                return [UnifiedEvent(
                    event_id=f"san_{eid}",
                    timestamp=datetime.now(timezone.utc),
                    ticker=ticker.upper(),
                    source_type=SourceType.SANTIMENT,
                    headline=f"Santiment Social Sentiment Index: {val:.1f}",
                    text=f"Aggregated social volume and sentiment polarity score: {score:+.2f}",
                    sentiment_label=(
                        SentimentLabel.BULLISH if score > 0.15
                        else SentimentLabel.BEARISH if score < -0.15
                        else SentimentLabel.NEUTRAL
                    ),
                    sentiment_score=round(score, 3),
                    impact_score=0.7,
                    urgency_weight=0.8,
                    metadata={"metric_value": val},
                )]

        res = await _santiment_breaker.call(_fetch, fallback=[])
        return res or []

    # ─── X (Twitter) Ingestion ───────────────────────────────────────────────

    async def fetch_twitter(self, ticker: str, limit: int = 10) -> List[UnifiedEvent]:
        """Fetch recent cashtag posts via Twitter / X API v2 search."""
        if not settings.twitter_bearer_token:
            return []

        async def _fetch() -> List[UnifiedEvent]:
            await _social_limiter.acquire()
            url = "https://api.twitter.com/2/tweets/search/recent"
            bearer = settings.twitter_bearer_token
            headers = {"Authorization": "Bearer " + bearer}
            params = {
                "query": f"${ticker} lang:en -is:retweet",
                "max_results": min(limit, 50),
                "tweet.fields": "created_at,public_metrics,author_id",
            }
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                tweets = data.get("data", [])
                events: List[UnifiedEvent] = []

                for t in tweets:
                    tid = t.get("id", "")
                    text = t.get("text", "")
                    metrics = t.get("public_metrics", {})
                    scored = await self.finbert.score_text(text)
                    score = scored["score"]
                    label = (
                        SentimentLabel.BULLISH if score > 0.15
                        else SentimentLabel.BEARISH if score < -0.15
                        else SentimentLabel.NEUTRAL
                    )
                    events.append(UnifiedEvent(
                        event_id=f"tw_{tid}",
                        timestamp=datetime.now(timezone.utc),
                        ticker=ticker.upper(),
                        source_type=SourceType.TWITTER,
                        headline=f"X post: {text[:60]}...",
                        text=text,
                        author=t.get("author_id"),
                        sentiment_label=label,
                        sentiment_score=score,
                        sentiment_probs=scored["probs"],
                        impact_score=scored["impact_score"],
                        urgency_weight=0.75,
                        metadata={
                            "retweets": metrics.get("retweet_count", 0),
                            "likes": metrics.get("like_count", 0),
                        },
                    ))
                return events

        res = await _twitter_breaker.call(_fetch, fallback=[])
        return res or []

    async def fetch_all_social(self, ticker: str) -> List[UnifiedEvent]:
        """Fetch concurrently from all social feeds and return combined events."""
        results = await asyncio.gather(
            self.fetch_stocktwits(ticker),
            self.fetch_reddit(ticker),
            self.fetch_santiment(ticker),
            self.fetch_twitter(ticker),
            return_exceptions=True,
        )
        combined: List[UnifiedEvent] = []
        for r in results:
            if isinstance(r, list):
                combined.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"Social fetcher error: {r}")
        return combined


@tool
def fetch_social_sentiment(ticker: str) -> list[dict]:
    """Fetch social sentiment events from StockTwits, Reddit, and available streams."""
    engine = SocialSentimentIngestionEngine()
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Fallback for sync execution in existing event loop
        return []
    events = asyncio.run(engine.fetch_all_social(ticker))
    return [e.to_dict() for e in events]
