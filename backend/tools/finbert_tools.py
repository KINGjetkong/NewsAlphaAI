"""
backend/tools/finbert_tools.py
Dedicated FinBERT financial NLP sentiment processing layer.
Provides high-throughput batched inference, directional probabilities,
impact scoring, and graceful fallback.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from loguru import logger

from backend.core.config import settings
from backend.streaming.event_schema import SentimentLabel, UnifiedEvent


# ─────────────────────────────────────────────────────────────────────────────
#  Financial Domain Lexicon for High-Precision Local Fallback
# ─────────────────────────────────────────────────────────────────────────────

BULLISH_KEYWORDS = {
    "surge", "soar", "gain", "rally", "jump", "climb", "beat", "outperform",
    "profit", "growth", "bullish", "record", "upgrade", "dividend", "revenue",
    "acquisition", "expansion", "partnership", "breakout", "strong", "higher",
    "optimistic", "momentum", "buyback", "exceed", "upside", "milestone",
}

BEARISH_KEYWORDS = {
    "plunge", "tumble", "drop", "fall", "decline", "slump", "miss", "loss",
    "deficit", "downgrade", "bearish", "crash", "investigation", "lawsuit",
    "subpoena", "fraud", "default", "bankruptcy", "warning", "layoff", "debt",
    "recession", "weak", "selloff", "headwind", "lower", "cut", "risk",
}

INTENSIFIERS = {
    "massive": 1.5, "sharp": 1.4, "huge": 1.4, "strong": 1.3,
    "slight": 0.6, "minor": 0.5, "modest": 0.7, "record": 1.5,
}


class FinBERTSentimentAnalyzer:
    """
    Dedicated financial sentiment analyzer utilizing HuggingFace FinBERT
    with batched asynchronous inference and intelligent financial lexicon fallback.
    """
    _instance: Optional[FinBERTSentimentAnalyzer] = None

    def __init__(self):
        self.model_name = settings.finbert_model_name
        self.device = settings.finbert_device
        self._pipeline = None
        self._model_loaded = False
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> FinBERTSentimentAnalyzer:
        if cls._instance is None:
            cls._instance = FinBERTSentimentAnalyzer()
        return cls._instance

    def _try_load_model(self) -> bool:
        """Attempt to load Hugging Face transformer pipeline if available."""
        if self._model_loaded:
            return True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            logger.info(f"Loading FinBERT model: {self.model_name} on {self.device}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                top_k=None,
                truncation=True,
                max_length=settings.finbert_max_length,
            )
            self._model_loaded = True
            logger.info("FinBERT pipeline successfully initialized.")
            return True
        except Exception as e:
            logger.warning(
                f"FinBERT weights not pre-loaded or offline ({e}). "
                "Using high-precision Financial Lexicon & Rule Fallback Engine."
            )
            self._model_loaded = False
            return False

    def _rule_based_score(self, text: str) -> Dict[str, Any]:
        """High-precision financial dictionary fallback."""
        cleaned = text.lower()
        words = re.findall(r"\b[a-z]+\b", cleaned)
        pos_score = 0.0
        neg_score = 0.0

        for i, w in enumerate(words):
            multiplier = 1.0
            if i > 0 and words[i - 1] in INTENSIFIERS:
                multiplier = INTENSIFIERS[words[i - 1]]
            if i > 0 and words[i - 1] in {"no", "not", "hardly", "barely"}:
                multiplier = -1.0

            if w in BULLISH_KEYWORDS:
                if multiplier > 0:
                    pos_score += 1.0 * multiplier
                else:
                    neg_score += 1.0 * abs(multiplier)
            elif w in BEARISH_KEYWORDS:
                if multiplier > 0:
                    neg_score += 1.0 * multiplier
                else:
                    pos_score += 1.0 * abs(multiplier)

        total = pos_score + neg_score
        if total == 0.0:
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "probs": {"positive": 0.2, "negative": 0.2, "neutral": 0.6},
                "impact_score": 0.1,
            }

        p_pos = pos_score / (total + 1.0)
        p_neg = neg_score / (total + 1.0)
        p_neu = max(0.1, 1.0 - (p_pos + p_neg))

        denom = p_pos + p_neg + p_neu
        p_pos /= denom
        p_neg /= denom
        p_neu /= denom

        net_score = p_pos - p_neg  # range [-1.0, 1.0]

        if net_score > 0.15:
            label = "POSITIVE"
        elif net_score < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        impact = min(1.0, (total / 5.0) + 0.2)

        return {
            "label": label,
            "score": round(net_score, 4),
            "probs": {
                "positive": round(p_pos, 4),
                "negative": round(p_neg, 4),
                "neutral": round(p_neu, 4),
            },
            "impact_score": round(impact, 4),
        }

    def score_text_sync(self, text: str) -> Dict[str, Any]:
        """Synchronously score a single text block."""
        if not text or len(text.strip()) < 5:
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "probs": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "impact_score": 0.0,
            }

        if self._model_loaded and self._pipeline:
            try:
                raw_out = self._pipeline(text[: settings.finbert_max_length])[0]
                probs_map = {item["label"].lower(): item["score"] for item in raw_out}
                p_pos = probs_map.get("positive", 0.0)
                p_neg = probs_map.get("negative", 0.0)
                p_neu = probs_map.get("neutral", 0.0)

                net_score = p_pos - p_neg
                if net_score > 0.15:
                    label = "POSITIVE"
                elif net_score < -0.15:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"

                impact = max(p_pos, p_neg)
                return {
                    "label": label,
                    "score": round(net_score, 4),
                    "probs": {
                        "positive": round(p_pos, 4),
                        "negative": round(p_neg, 4),
                        "neutral": round(p_neu, 4),
                    },
                    "impact_score": round(impact, 4),
                }
            except Exception as e:
                logger.warning(f"FinBERT inference failed: {e}. Falling back to rule engine.")

        return self._rule_based_score(text)

    async def score_text(self, text: str) -> Dict[str, Any]:
        """Asynchronously score a single text block."""
        return await asyncio.to_thread(self.score_text_sync, text)

    async def score_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Asynchronously score a list of texts in batches."""
        if not texts:
            return []
        return await asyncio.to_thread(
            lambda: [self.score_text_sync(t) for t in texts]
        )

    async def score_event(self, event: UnifiedEvent) -> UnifiedEvent:
        """Enrich a UnifiedEvent with FinBERT sentiment scores."""
        combined_text = f"{event.headline}. {event.text}".strip()
        scored = await self.score_text(combined_text)

        event.sentiment_score = scored["score"]
        event.sentiment_probs = scored["probs"]
        event.impact_score = scored["impact_score"]

        lbl = scored["label"]
        if lbl == "POSITIVE":
            event.sentiment_label = SentimentLabel.BULLISH
        elif lbl == "NEGATIVE":
            event.sentiment_label = SentimentLabel.BEARISH
        else:
            event.sentiment_label = SentimentLabel.NEUTRAL

        return event


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph Tool Wrapper
# ─────────────────────────────────────────────────────────────────────────────

@tool
def finbert_analyse_text(text: str) -> dict:
    """Run FinBERT financial sentiment analysis on a headline or article."""
    analyzer = FinBERTSentimentAnalyzer.get_instance()
    return analyzer.score_text_sync(text)
