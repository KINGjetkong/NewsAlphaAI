"""
backend/api/main.py
FastAPI application with REST endpoints + WebSocket for real-time pipeline streaming.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.core.config import settings
from backend.api.lifespan import lifespan
from backend.pipelines.prediction_pipeline import run_prediction
from backend.rl.feedback_loop import RLFeedbackLoop
from backend.memory.memory_manager import AgentMemoryManager
from backend.tools.stock_tools import (
    get_stock_info,
    compute_technical_features,
    fetch_stock_ohlcv,
)
from backend.streaming.event_schema import MarketTick, UnifiedEvent
from backend.streaming.deephaven_engine import get_deephaven_engine
from backend.streaming.cross_modal_linker import get_cross_modal_linker
from backend.streaming.event_broker import get_event_broker
from backend.ml.regime_model import HiddenMarkovRegimeChecker
from backend.tools.news_tools import AsyncNewsEngine
from backend.tools.trends_tools import GoogleTrendsTracker
from backend.tools.social_sentiment_tools import SocialSentimentIngestionEngine
from backend.tools.flow_political_tools import FlowAndPoliticalEngine


app = FastAPI(
    title="NewsAlphaAI API",
    description="AI-powered stock prediction via news analysis + LLM + RL",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _safe_send(websocket: WebSocket, payload: dict) -> bool:
    """Send JSON safely. Returns False if the socket is already closed."""
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  REST Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/stock/{ticker}/info")
async def stock_info(ticker: str):
    """Basic company info."""
    return get_stock_info.invoke({"ticker": ticker.upper()})


@app.get("/api/stock/{ticker}/chart")
async def stock_chart(ticker: str, period: str = "30d"):
    """OHLCV data for chart rendering."""
    return fetch_stock_ohlcv.invoke({"ticker": ticker.upper(), "period": period})


@app.get("/api/stock/{ticker}/technicals")
async def stock_technicals(ticker: str):
    """Technical indicator snapshot."""
    return compute_technical_features.invoke({"ticker": ticker.upper()})


@app.get("/api/prediction/{ticker}/history")
async def prediction_history(ticker: str):
    """Return historical predictions from memory."""
    mgr = AgentMemoryManager(ticker.upper())
    return {"ticker": ticker.upper(), "history": mgr.get_prediction_history()}


@app.get("/api/prediction/{ticker}/rl-stats")
async def rl_stats(ticker: str):
    """RL model accuracy stats."""
    rl = RLFeedbackLoop()
    return rl.get_stats(ticker.upper())


@app.post("/api/predict/{ticker}")
async def predict_rest(ticker: str):
    """
    Run full prediction pipeline synchronously (blocking).
    For real-time streaming, use the WebSocket endpoint instead.
    """
    try:
        state = await run_prediction(ticker.upper())
        return state.get("final_prediction", {})
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/deephaven/{ticker}/snapshot")
async def deephaven_snapshot(ticker: str):
    """
    Return recent As-Of joined ticks, sentiment velocity, and cross-modal clusters.
    """
    ticker_sym = ticker.upper()
    engine = get_deephaven_engine()
    linker = get_cross_modal_linker()
    hmm = HiddenMarkovRegimeChecker()

    recent_ticks = engine.get_recent_synchronized_ticks(ticker_sym, limit=50)
    velocity = linker.compute_sentiment_velocity(ticker_sym)
    clusters = linker.find_cross_modal_clusters(ticker_sym)

    # Compute regime snapshot
    regime = hmm.assess_regime(
        ticker=ticker_sym,
        price_change_pct=0.0,
        volatility_pct=1.5,
        volume_ratio=1.0,
        sentiment_velocity=velocity.sentiment_velocity,
    )

    return {
        "ticker": ticker_sym,
        "synchronized_ticks": [t.to_dict() for t in recent_ticks],
        "sentiment_velocity": velocity.model_dump(),
        "cross_modal_clusters": [c.model_dump() for c in clusters],
        "market_regime": regime.model_dump(),
    }


@app.get("/api/sentiment/multi-source/{ticker}")
async def fetch_multi_source_sentiment(ticker: str):
    """
    Concurrently fetch alternative data across News/SEC, Google Trends,
    Social Sentiment, and Institutional Flow with FinBERT scoring.
    """
    ticker_sym = ticker.upper()
    news_engine = AsyncNewsEngine()
    trends_engine = GoogleTrendsTracker()
    social_engine = SocialSentimentIngestionEngine()
    flow_engine = FlowAndPoliticalEngine()

    results = await asyncio.gather(
        news_engine.fetch_all_events_async(ticker_sym),
        trends_engine.fetch_trends_events(ticker_sym),
        social_engine.fetch_all_social(ticker_sym),
        flow_engine.fetch_all_flow_and_political(ticker_sym),
        return_exceptions=True,
    )

    all_events: list[UnifiedEvent] = []
    for r in results:
        if isinstance(r, list):
            all_events.extend(r)
        elif isinstance(r, Exception):
            logger.warning(f"Error fetching alternative data for {ticker_sym}: {r}")

    # Ingest into Deephaven sync engine and CrossModalLinker
    engine = get_deephaven_engine()
    linker = get_cross_modal_linker()
    for ev in all_events:
        engine.ingest_unified_event(ev)
        linker.ingest_event(ev)

    velocity = linker.compute_sentiment_velocity(ticker_sym)
    clusters = linker.find_cross_modal_clusters(ticker_sym)

    return {
        "ticker": ticker_sym,
        "event_count": len(all_events),
        "events": [e.to_dict() for e in all_events],
        "sentiment_velocity": velocity.model_dump(),
        "cross_modal_clusters": [c.model_dump() for c in clusters],
    }


@app.get("/api/deephaven/{ticker}/displacement")
async def deephaven_displacement(ticker: str, event_id: str, forward_ticks: int = 50):
    """Return post-shock tick-by-tick displacement series."""
    engine = get_deephaven_engine()
    series = engine.get_displacement_series(ticker.upper(), event_id, forward_ticks=forward_ticks)
    return {
        "ticker": ticker.upper(),
        "event_id": event_id,
        "displacement_series": series,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket – Real-time pipeline streaming
# ─────────────────────────────────────────────────────────────────────────────

STEP_LABELS = {
    "fetch_news":        "Fetching news articles...",
    "analyse_sentiment": "Analysing sentiment with LLM...",
    "embed_store":       "Embedding and storing in vector DB...",
    "fetch_stock":       "Fetching stock data and technicals...",
    "rag_retrieve":      "RAG retrieval from vector store...",
    "llm_reason":        "LLM reasoning about price direction...",
    "ml_predict":        "ML model prediction...",
    "final_predict":     "Generating final ensemble prediction...",
}


@app.websocket("/ws/predict/{ticker}")
async def ws_predict(websocket: WebSocket, ticker: str):
    """
    WebSocket endpoint that streams pipeline progress to the frontend.
    Message types:
      start       — pipeline is starting
      step_start  — a node just began executing
      step        — a node just finished executing
      complete    — all nodes done, final prediction attached
      error       — something went wrong
    """
    await websocket.accept()
    ticker = ticker.upper()
    logger.info(f"WS predict: connected for {ticker}")
    connected = True

    try:
        sent = await _safe_send(websocket, {
            "type": "start",
            "ticker": ticker,
            "message": f"Starting prediction pipeline for {ticker}",
            "timestamp": datetime.utcnow().isoformat(),
        })
        if not sent:
            return

        async def on_step(step_name: str, state: dict):
            if not connected:
                return

            # __start__X  →  node just began, send step_start
            if step_name.startswith("__start__"):
                actual = step_name.replace("__start__", "")
                await _safe_send(websocket, {
                    "type": "step_start",
                    "step": actual,
                    "label": STEP_LABELS.get(actual, actual),
                    "logs": state.get("logs", [])[-2:],
                    "timestamp": datetime.utcnow().isoformat(),
                })
                return

            # Regular name  →  node just finished, send step
            payload = {
                "type": "step",
                "step": step_name,
                "label": STEP_LABELS.get(step_name, step_name),
                "logs": state.get("logs", [])[-3:],
                "timestamp": datetime.utcnow().isoformat(),
            }
            if step_name == "fetch_news":
                payload["article_count"] = len(state.get("raw_articles", []))
            elif step_name == "analyse_sentiment":
                payload["sentiment"] = state.get("batch_sentiment", {})
            elif step_name == "fetch_stock":
                payload["technicals"] = state.get("tech_features", {})
            elif step_name == "final_predict":
                payload["prediction"] = state.get("final_prediction", {})

            await _safe_send(websocket, payload)

        final_state = await run_prediction(ticker, on_step=on_step)

        await _safe_send(websocket, {
            "type": "complete",
            "ticker": ticker,
            "prediction": final_state.get("final_prediction", {}),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except WebSocketDisconnect:
        logger.info(f"WS predict: client disconnected for {ticker}")
    except Exception as e:
        logger.error(f"WS predict error for {ticker}: {e}")
        await _safe_send(websocket, {"type": "error", "message": str(e)})
    finally:
        connected = False
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"WS predict: closed for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket – Live market ticker broadcast (price updates)
# ─────────────────────────────────────────────────────────────────────────────

_price_subscribers: dict[str, list[WebSocket]] = {}


@app.websocket("/ws/live/{ticker}")
async def ws_live_price(websocket: WebSocket, ticker: str):
    """
    Broadcasts price updates every 15s for the given ticker.
    Multiple clients can subscribe to the same ticker.
    """
    await websocket.accept()
    ticker = ticker.upper()
    _price_subscribers.setdefault(ticker, []).append(websocket)
    logger.info(f"WS live: connected for {ticker}")

    try:
        import yfinance as yf
        while True:
            try:
                hist = yf.Ticker(ticker).history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                    change = price - prev
                    change_pct = change / (prev + 1e-9) * 100
                    sent = await _safe_send(websocket, {
                        "type": "price",
                        "ticker": ticker,
                        "price": round(price, 4),
                        "change": round(change, 4),
                        "change_pct": round(change_pct, 4),
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    if not sent:
                        break
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception as e:
                logger.warning(f"WS live price fetch error for {ticker}: {e}")
                sent = await _safe_send(websocket, {
                    "type": "error",
                    "message": str(e),
                })
                if not sent:
                    break

            await asyncio.sleep(15)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS live unexpected error for {ticker}: {e}")
    finally:
        if ticker in _price_subscribers:
            _price_subscribers[ticker] = [
                ws for ws in _price_subscribers[ticker] if ws != websocket
            ]
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"WS live: closed for {ticker}")


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket – Deephaven Tick-by-Tick Synchronization & Alternative Data Stream
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/deephaven/{ticker}")
async def ws_deephaven_sync(websocket: WebSocket, ticker: str):
    """
    Real-time streaming WebSocket synchronizing sub-second market ticks with
    FinBERT multi-source sentiment, options flow, and Google Trends momentum.
    """
    await websocket.accept()
    ticker = ticker.upper()
    logger.info(f"WS Deephaven: connected for {ticker}")

    engine = get_deephaven_engine()
    linker = get_cross_modal_linker()
    broker = get_event_broker()

    # Send initial snapshot
    init_ticks = [t.to_dict() for t in engine.get_recent_synchronized_ticks(ticker, limit=20)]
    init_velocity = linker.compute_sentiment_velocity(ticker).model_dump()
    init_clusters = [c.model_dump() for c in linker.find_cross_modal_clusters(ticker)]

    await _safe_send(websocket, {
        "type": "deephaven_init",
        "ticker": ticker,
        "synchronized_ticks": init_ticks,
        "sentiment_velocity": init_velocity,
        "cross_modal_clusters": init_clusters,
    })

    # Queue for delivering broadcast messages to this client
    send_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    def on_asof_tick(joined_tick):
        if hasattr(joined_tick, "ticker") and joined_tick.ticker == ticker:
            try:
                send_queue.put_nowait({
                    "type": "asof_tick",
                    "data": joined_tick.to_dict(),
                })
            except asyncio.QueueFull:
                pass

    def on_unified_event(event):
        if hasattr(event, "ticker") and event.ticker == ticker:
            try:
                send_queue.put_nowait({
                    "type": "unified_event",
                    "data": event.to_dict(),
                })
            except asyncio.QueueFull:
                pass

    broker.subscribe("asof_tick", on_asof_tick)
    broker.subscribe("unified_event", on_unified_event)

    async def client_writer():
        while True:
            msg = await send_queue.get()
            ok = await _safe_send(websocket, msg)
            if not ok:
                break

    async def client_reader():
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                if msg_type == "tick":
                    # Ingest custom incoming market tick
                    price = float(data.get("price", 0.0))
                    size = float(data.get("size", 100))
                    bid = float(data.get("bid", price - 0.01))
                    ask = float(data.get("ask", price + 0.01))
                    tick_obj = MarketTick(
                        tick_id=f"tick_{datetime.now(timezone.utc).strftime('%H%M%S%f')[:10]}",
                        timestamp=datetime.now(timezone.utc),
                        ticker=ticker,
                        price=price,
                        size=size,
                        bid=bid,
                        ask=ask,
                    )
                    joined = engine.ingest_market_tick(tick_obj)
                    await _safe_send(websocket, {
                        "type": "asof_tick",
                        "data": joined.to_dict(),
                    })
                elif msg_type == "ping":
                    await _safe_send(websocket, {"type": "pong"})
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception as e:
                logger.warning(f"WS Deephaven read error: {e}")
                break

    writer_task = asyncio.create_task(client_writer())
    reader_task = asyncio.create_task(client_reader())

    try:
        done, pending = await asyncio.wait(
            [writer_task, reader_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        broker.unsubscribe("asof_tick", on_asof_tick)
        broker.unsubscribe("unified_event", on_unified_event)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"WS Deephaven: closed for {ticker}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
