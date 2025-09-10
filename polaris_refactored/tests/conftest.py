import asyncio
import json
import pytest
from typing import Any, Callable, Dict, List


class FakeMessageBroker:
    """In-memory broker used for testing PolarisMessageBus.

    - Tracks connect/disconnect state
    - Stores subscriptions by topic -> callback
    - publish() immediately invokes the subscribed callback(s)
    """

    def __init__(self) -> None:
        self.connected: bool = False
        self.published: List[Dict[str, Any]] = []
        self._subs: Dict[str, Callable[[bytes], None]] = {}
        self._id_counter = 0

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        self._subs.clear()

    async def publish(self, topic: str, message: bytes) -> None:
        # record for assertions
        self.published.append({"topic": topic, "message": message})
        # deliver to subscriber if present
        cb = self._subs.get(topic)
        if cb:
            # emulate async callback invocation
            await asyncio.sleep(0)
            cb(message)

    async def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> str:
        self._id_counter += 1
        sub_id = f"sub-{self._id_counter}"
        # only one subscription per topic for simplicity; message_bus multiplexes handlers per topic itself
        self._subs[topic] = handler
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        # Remove any topic that used this subscription id (we don't map id->topic in this simple fake)
        # For test purposes, clear all subs to simulate broker unsubscribed for the topic requested by the bus
        # The bus removes the mapping on its side, so we can be lenient here.
        return None


class FakeMetricsCollector:
    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}

    def increment_counter(self, name: str, labels: Dict[str, str]) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1


@pytest.fixture
def fake_broker():
    return FakeMessageBroker()


@pytest.fixture
def metrics_collector():
    return FakeMetricsCollector()
