"""
backend/api/lifespan.py
FastAPI lifespan context manager – starts/stops the APScheduler.
Import and attach to the FastAPI app in main.py.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from loguru import logger
from fastapi import FastAPI

from backend.utils.scheduler import create_scheduler

_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    logger.info("Starting background scheduler…")
    _scheduler = create_scheduler()
    _scheduler.start()

    from backend.streaming.event_broker import get_event_broker
    from backend.streaming.deephaven_engine import get_deephaven_engine

    broker = get_event_broker()
    await broker.start()
    get_deephaven_engine()
    logger.info("Asynchronous event broker & Deephaven sync engine active.")

    yield

    logger.info("Shutting down event broker and scheduler…")
    await broker.stop()
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
