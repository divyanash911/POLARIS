import asyncio
import pytest
from datetime import datetime
from infrastructure.message_bus import PolarisMessageBus

# Mocking missing classes for tests if needed, or removing dependence.
# The current PolarisMessageBus implementation does not have middleware_chain attribute.

@pytest.mark.asyncio
async def test_publish_and_subscribe(fake_broker):  # fake_broker is largely unused by the simplified bus but kept if fixture requires it
    bus = PolarisMessageBus()
    await bus.start()

    received_events = []
    
    async def handler(message):
        received_events.append(message)

    bus.subscribe("test.topic", handler)
    
    await bus.publish("test.topic", {"data": 123})
    
    # Allow some time for processing
    await asyncio.sleep(0.1)
    
    assert len(received_events) == 1
    assert received_events[0].topic == "test.topic"
    assert received_events[0].payload == {"data": 123}
    
    await bus.stop()

@pytest.mark.asyncio
async def test_unsubscribe():
    bus = PolarisMessageBus()
    await bus.start()
    
    received_events = []
    async def handler(message):
        received_events.append(message)
        
    bus.subscribe("test.topic", handler)
    bus.unsubscribe("test.topic", handler)
    
    await bus.publish("test.topic", {"data": 456})
    await asyncio.sleep(0.1)
    
    assert len(received_events) == 0
    await bus.stop()
