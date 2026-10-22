"""
backend/ml/regime_model.py
Hidden Markov Model (HMM) market regime checker and multi-horizon outcome evaluator.
Provides market state identification (Bull Momentum, Bear Distribution, Volatility Chop,
Consolidation) and tilt modifiers for downstream ML & execution algorithms.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class MarketRegime(str, Enum):
    BULL_EXPANSION = "BULL_EXPANSION"
    BEAR_DISTRIBUTION = "BEAR_DISTRIBUTION"
    HIGH_VOLATILITY_CHOP = "HIGH_VOLATILITY_CHOP"
    LOW_VOL_CONSOLIDATION = "LOW_VOL_CONSOLIDATION"


class RegimeAssessment(BaseModel):
    ticker: str
    regime: MarketRegime
    confidence: float
    state_probabilities: Dict[str, float]
    tilt_multiplier: float
    suggested_action: str
    risk_haircut: float


class HiddenMarkovRegimeChecker:
    """
    4-State Markov Regime Switching model estimating current latent market state
    from price returns, realized volatility, volume ratios, and sentiment velocity.
    """

    def __init__(self) -> None:
        # Default Transition Probability Matrix A[i, j] = P(State_t = j | State_{t-1} = i)
        # States: 0=BULL_EXPANSION, 1=BEAR_DISTRIBUTION, 2=HIGH_VOL_CHOP, 3=LOW_VOL_CONSOLIDATION
        self.transition_matrix = [
            [0.85, 0.05, 0.07, 0.03],  # From Bull
            [0.05, 0.85, 0.08, 0.02],  # From Bear
            [0.10, 0.12, 0.70, 0.08],  # From High Vol Chop
            [0.15, 0.10, 0.05, 0.70],  # From Low Vol
        ]
        # Prior state probabilities
        self.state_priors = [0.35, 0.25, 0.20, 0.20]

    def assess_regime(
        self,
        ticker: str,
        price_change_pct: float,
        volatility_pct: float,
        volume_ratio: float,
        sentiment_velocity: float = 0.0,
        flow_imbalance: float = 0.0,
    ) -> RegimeAssessment:
        """
        Estimate emission probabilities and compute posterior distribution over regimes.
        """
        # Observation likelihoods under each state
        # 0: Bull Expansion -> positive returns, normal vol, positive flow and sentiment
        p_bull = (
            self._gaussian_pdf(price_change_pct, mean=1.2, std=1.0)
            * self._gaussian_pdf(sentiment_velocity, mean=0.3, std=0.4)
            * (1.2 if flow_imbalance > 0 else 0.8)
        )

        # 1: Bear Distribution -> negative returns, elevated vol, negative flow & sentiment
        p_bear = (
            self._gaussian_pdf(price_change_pct, mean=-1.5, std=1.2)
            * self._gaussian_pdf(sentiment_velocity, mean=-0.3, std=0.4)
            * (1.2 if flow_imbalance < 0 else 0.8)
        )

        # 2: High Volatility Chop -> large absolute returns, high vol, neutral flow
        p_chop = (
            self._gaussian_pdf(abs(price_change_pct), mean=2.5, std=1.5)
            * (1.5 if volatility_pct > 2.0 else 0.7)
        )

        # 3: Low Vol Consolidation -> near-zero price change, low vol, low volume ratio
        p_consol = (
            self._gaussian_pdf(abs(price_change_pct), mean=0.1, std=0.4)
            * (1.4 if volume_ratio < 0.9 else 0.8)
        )

        raw_probs = [
            self.state_priors[0] * p_bull,
            self.state_priors[1] * p_bear,
            self.state_priors[2] * p_chop,
            self.state_priors[3] * p_consol,
        ]

        total = sum(raw_probs)
        if total <= 1e-9:
            norm_probs = [0.25, 0.25, 0.25, 0.25]
        else:
            norm_probs = [p / total for p in raw_probs]

        prob_map = {
            MarketRegime.BULL_EXPANSION.value: round(norm_probs[0], 3),
            MarketRegime.BEAR_DISTRIBUTION.value: round(norm_probs[1], 3),
            MarketRegime.HIGH_VOLATILITY_CHOP.value: round(norm_probs[2], 3),
            MarketRegime.LOW_VOL_CONSOLIDATION.value: round(norm_probs[3], 3),
        }

        # Select maximum likelihood regime
        regimes = [
            MarketRegime.BULL_EXPANSION,
            MarketRegime.BEAR_DISTRIBUTION,
            MarketRegime.HIGH_VOLATILITY_CHOP,
            MarketRegime.LOW_VOL_CONSOLIDATION,
        ]
        best_idx = norm_probs.index(max(norm_probs))
        regime = regimes[best_idx]
        conf = round(norm_probs[best_idx], 3)

        # Compute tilt multiplier (-1.0 to +1.0) and risk haircut
        if regime == MarketRegime.BULL_EXPANSION:
            tilt = round(0.4 + 0.6 * conf, 2)
            haircut = 0.0
            action = "AGGRESSIVE_LONG_OR_MOMENTUM"
        elif regime == MarketRegime.BEAR_DISTRIBUTION:
            tilt = round(-0.4 - 0.6 * conf, 2)
            haircut = 0.2
            action = "DEFENSIVE_SHORT_OR_CASH"
        elif regime == MarketRegime.HIGH_VOLATILITY_CHOP:
            tilt = round(0.1 * (norm_probs[0] - norm_probs[1]), 2)
            haircut = 0.5
            action = "REDUCE_SIZE_MEAN_REVERSION"
        else:
            tilt = 0.0
            haircut = 0.1
            action = "WAIT_FOR_BREAKOUT"

        return RegimeAssessment(
            ticker=ticker.upper(),
            regime=regime,
            confidence=conf,
            state_probabilities=prob_map,
            tilt_multiplier=tilt,
            suggested_action=action,
            risk_haircut=haircut,
        )

    @staticmethod
    def _gaussian_pdf(x: float, mean: float, std: float) -> float:
        if std <= 0:
            std = 1.0
        exp_term = math.exp(-0.5 * ((x - mean) / std) ** 2)
        return max(1e-6, (1.0 / (std * math.sqrt(2 * math.pi))) * exp_term)


class MultiHorizonEvaluator:
    """
    Evaluates ML predictions and sentiment shocks across multiple time horizons:
    1-minute, 5-minute, 1-hour, 1-day, and 3-day return outcomes.
    """

    HORIZONS = {
        "1m": 60,
        "5m": 300,
        "1h": 3600,
        "1d": 86400,
        "3d": 259200,
    }

    @staticmethod
    def compute_outcomes(
        entry_price: float,
        horizon_prices: Dict[str, float],
        predicted_direction: str,
    ) -> Dict[str, Any]:
        """Compare predicted direction against actual returns across horizons."""
        results = {}
        target_sign = 1 if predicted_direction.upper() == "UP" else -1

        for h, price in horizon_prices.items():
            if price and entry_price > 0:
                ret_bps = ((price - entry_price) / entry_price) * 10000.0
                actual_sign = 1 if ret_bps > 0 else (-1 if ret_bps < 0 else 0)
                correct = (target_sign == actual_sign)
                results[h] = {
                    "horizon_price": round(price, 4),
                    "return_bps": round(ret_bps, 2),
                    "correct": correct,
                }
            else:
                results[h] = {"return_bps": None, "correct": None}

        return {
            "entry_price": entry_price,
            "predicted_direction": predicted_direction,
            "horizons": results,
        }
