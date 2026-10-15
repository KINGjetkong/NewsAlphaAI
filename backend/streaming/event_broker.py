"""
backend/streaming/event_broker.py
Asynchronous pub/sub event bus supporting in-memory queues and Redis/ZMQ brokers.
Decouples upstream data ingestion from downstream NLP processing and Deephaven sync.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from loguru import logger

from backend.streaming.event_schema import MarketTick, UnifiedEvent


class EventBroker:
    """
    Central reactive event bus. Allows concurrent workers to publish UnifiedEvents
    and MarketTicks, and subscribers (WebSocket clients, Deephaven engine, ML feeders)
    to consume real-time streams without blocking.
    """
    _instance: Optional[EventBroker] = None

    def __init__(self, max_history: int = 500):
        # channel -> set of asyncio.Queue
        self._subscribers: Dict[str, set[asyncio.Queue]] = defaultdict(set)
        # ticker -> ring buffer of recent UnifiedEvents
        self._event_history: Dict[str, deque[UnifiedEvent]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        # ticker -> ring buffer of recent MarketTicks
        self._tick_history: Dict[str, deque[MarketTick]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._lock = asyncio.Lock()
        self._running_tasks: List[asyncio.Task] = []

    @classmethod
    def get_instance(cls) -> EventBroker:
        if cls._instance is None:
            cls._instance = EventBroker()
        return cls._instance

    # ─── Subscription ────────────────────────────────────────────────────────

    async def subscribe(self, channel: str) -> asyncio.Queue:
        """Register a new consumer queue on a specific channel."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._subscribers[channel].add(queue)
        logger.debug(f"[Broker] New subscriber on channel: {channel}")
        return queue

    async def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        """Remove consumer queue upon disconnection."""
        async with self._lock:
            if channel in self._subscribers:
                self._subscribers[channel].discard(queue)
                if not self._subscribers[channel]:
                    del self._subscribers[channel]
        logger.debug(f"[Broker] Unsubscribed from channel: {channel}")

    # ─── Publishing ──────────────────────────────────────────────────────────

    async def publish_event(self, event: UnifiedEvent) -> None:
        """Publish a UnifiedEvent to ticker-specific and global channels."""
        ticker = event.ticker.upper()
        self._event_history[ticker].append(event)
        self._event_history["ALL"].append(event)

        channels = [f"events:{ticker}", "events:ALL"]
        payload = event.to_dict()

        for ch in channels:
            queues = list(self._subscribers.get(ch, set()))
            for q in queues:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    # Drop oldest if congested to avoid blocking producer
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    q.put_nowait(payload)

    async def publish_tick(self, tick: MarketTick) -> None:
        """Publish a MarketTick to ticker-specific and global channels."""
        ticker = tick.ticker.upper()
        self._tick_history[ticker].append(tick)
        self._tick_history["ALL"].append(tick)

        channels = [f"ticks:{ticker}", "ticks:ALL"]
        payload = tick.to_dict()

        for ch in channels:
            queues = list(self._subscribers.get(ch, set()))
            for q in queues:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    q.put_nowait(payload)

    # ─── History Retrieval ───────────────────────────────────────────────────

    def get_recent_events(
        self, ticker: str, limit: int = 50
    ) -> List[UnifiedEvent]:
        history = self._event_history.get(ticker.upper(), deque())
        items = list(history)
        return items[-limit:]

    def get_recent_ticks(
        self, ticker: str, limit: int = 50
    ) -> List[MarketTick]:
        history = self._tick_history.get(ticker.upper(), deque())
        items = list(history)
        return items[-limit:]

    # ─── Stream Generator Helper ─────────────────────────────────────────────

    async def event_stream(
        self, channel: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Convenience generator for streaming updates to SSE or WebSockets."""
        q = await self.subscribe(channel)
        try:
            while True:
                item = await q.get()
                yield item
                q.task_done()
        finally:
            await self.unsubscribe(channel, q)

    # ─── Worker Management ───────────────────────────────────────────────────

    def register_worker(self, coro: Callable[[], Any]) -> asyncio.Task:
        """Launch and track a background worker task."""
        task = asyncio.create_task(coro)
        self._running_tasks.append(task)
        return task

    async def shutdown(self) -> None:
        """Gracefully cancel and terminate background workers."""
        logger.info("[Broker] Shutting down event broker...")
        for task in self._running_tasks:
            if not task.done():
                task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
        self._subscribers.clear()
