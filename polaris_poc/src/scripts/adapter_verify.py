#!/usr/bin/env python3
"""
Verification script for SWIM adapters (Monitor + Execution)

- Monitor: listens to polaris.telemetry.events and polaris.telemetry.events.batch
- Execution: sends control actions to polaris.actions.swim_adapter and
             listens for polaris.execution.results and polaris.execution.metrics
"""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Set

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg

# -------------------- Utilities --------------------

def now_iso() -> str:
    return datetime.now(UTC).isoformat() + "Z"

# -------------------- Base Tester --------------------

class AdapterTester:
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc: Optional[NATS] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    async def connect(self):
        self.nc = NATS()
        await self.nc.connect(self.nats_url)
        self.logger.info(f"Connected to NATS at {self.nats_url}")

    async def close(self):
        if self.nc:
            await self.nc.close()
            self.logger.info("Closed NATS connection")


# -------------------- Monitor Adapter Tester --------------------

class MonitorAdapterTester(AdapterTester):
    """Tester for SWIM Monitor Adapter (handles single and batch telemetry)."""

    async def listen_to_telemetry(self, duration: int = 20) -> List[Dict[str, Any]]:
        await self.connect()
        events: List[Dict[str, Any]] = []

        async def single_handler(msg: Msg):
            try:
                data = json.loads(msg.data.decode())
                # New monitor uses 'name', 'value', 'unit', 'ts', 'source'
                events.append(data)
                self.logger.info(f"[telemetry.single] {data.get('name')} = {data.get('value')} {data.get('unit')} @ {data.get('ts')}")
            except Exception as e:
                self.logger.error("Error parsing single telemetry: %s", e)

        async def batch_handler(msg: Msg):
            try:
                batch = json.loads(msg.data.decode())
                batch_ts = batch.get("batch_ts", None)
                evts = batch.get("events", []) or []
                self.logger.info(f"[telemetry.batch] received batch of {len(evts)} events @ {batch_ts}")
                events.extend(evts)
            except Exception as e:
                self.logger.error("Error parsing batch telemetry: %s", e)

        # Subscribe to both subjects
        assert self.nc is not None
        await self.nc.subscribe("polaris.telemetry.events", cb=single_handler)
        await self.nc.subscribe("polaris.telemetry.events.batch", cb=batch_handler)

        self.logger.info("Listening for telemetry on 'polaris.telemetry.events' and 'polaris.telemetry.events.batch' for %s seconds", duration)
        await asyncio.sleep(duration)

        await self.close()
        self.logger.info("Finished listening to telemetry. Collected %d events", len(events))
        return events

    def verify_event_structure(self, events: List[Dict[str, Any]]) -> bool:
        if not events:
            self.logger.warning("No telemetry events to verify.")
            return False

        required = ["name", "value", "unit", "ts", "source"]
        for i, e in enumerate(events):
            for k in required:
                if k not in e:
                    self.logger.error("Event %d missing field '%s': %s", i, k, e)
                    return False
            if not isinstance(e["value"], (int, float)):
                # Attempt conversion if it's string numeric
                try:
                    float(e["value"])
                except Exception:
                    self.logger.error("Event %d value not numeric: %r", i, e["value"])
                    return False
        self.logger.info("Telemetry events structure OK (checked %d events)", len(events))
        return True


# -------------------- Execution Adapter Tester --------------------

class ExecutionAdapterTester(AdapterTester):
    """
    Tester for the SWIM Execution Adapter.

    - Publishes actions to 'polaris.actions.swim_adapter'
    - Subscribes to 'polaris.execution.results' and 'polaris.execution.metrics'
    - Correlates results by action_id
    """

    def __init__(self, nats_url: str = "nats://localhost:4222", result_timeout: float = 20.0):
        super().__init__(nats_url=nats_url)
        self.result_timeout = result_timeout
        self.results: List[Dict[str, Any]] = []
        self.metrics: List[Dict[str, Any]] = []
        self._pending_action_ids: Set[str] = set()
        self._result_by_action: Dict[str, Dict[str, Any]] = {}

    def create_test_actions(self) -> List[Dict[str, Any]]:
        ts = now_iso()
        actions = [
            {"action_type": "ADD_SERVER", "timestamp": ts, "source": "verify_script", "params": {}, "action_id": str(uuid.uuid4())},
            {"action_type": "SET_DIMMER", "timestamp": ts, "source": "verify_script", "params": {"value": 0.8}, "action_id": str(uuid.uuid4())},
            {"action_type": "ADJUST_QOS", "timestamp": ts, "source": "verify_script", "params": {"value": 0.5}, "action_id": str(uuid.uuid4())},
            {"action_type": "REMOVE_SERVER", "timestamp": ts, "source": "verify_script", "params": {}, "action_id": str(uuid.uuid4())},
        ]
        return actions

    async def test_action_execution(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send actions and wait for correlated execution results and some metrics."""

        await self.connect()
        assert self.nc is not None

        # Handlers
        async def result_handler(msg: Msg):
            try:
                data = json.loads(msg.data.decode())
                self.results.append(data)
                aid = data.get("action_id")
                if aid:
                    self._result_by_action[aid] = data
                    if aid in self._pending_action_ids:
                        self._pending_action_ids.remove(aid)
                self.logger.info("[result] %s -> %s (action_id=%s)", data.get("action_type"), "SUCCESS" if data.get("success") else "FAILED", aid)
            except Exception as e:
                self.logger.error("Error parsing result: %s", e)

        async def metrics_handler(msg: Msg):
            try:
                data = json.loads(msg.data.decode())
                self.metrics.append(data)
                self.logger.info("[metric] %s @ %s", data.get("metric"), data.get("ts"))
            except Exception as e:
                self.logger.error("Error parsing metric: %s", e)

        # Subscribe
        await self.nc.subscribe("polaris.execution.results", cb=result_handler)
        await self.nc.subscribe("polaris.execution.metrics", cb=metrics_handler)

        # Publish actions
        self._pending_action_ids = {a["action_id"] for a in actions}
        self.logger.info("Publishing %d actions (action_ids: %s)", len(actions), ", ".join(sorted(self._pending_action_ids)))

        for action in actions:
            # publish to 'polaris.actions.swim_adapter' (adapter subscribes also to polaris.actions.swim)
            await self.nc.publish("polaris.actions.swim_adapter", json.dumps(action).encode())
            # small inter-action pause to let adapter pickup in order
            await asyncio.sleep(0.2)

        # Wait until all action results received or timeout
        timeout = self.result_timeout + max(0, 2 * len(actions))
        waited = 0.0
        poll_interval = 0.25
        while waited < timeout and self._pending_action_ids:
            await asyncio.sleep(poll_interval)
            waited += poll_interval

        # If any pending remain, log them
        if self._pending_action_ids:
            self.logger.warning("Timeout waiting for results for %d actions: %s", len(self._pending_action_ids), ", ".join(sorted(self._pending_action_ids)))

        # Small extra wait to capture any trailing metrics
        await asyncio.sleep(1.0)

        await self.close()

        return {
            "sent_actions": actions,
            "results": self.results,
            "metrics": self.metrics,
            "result_by_action": self._result_by_action,
            "pending_action_ids": list(self._pending_action_ids),
        }


# -------------------- Test runners / CLI --------------------

async def test_monitor_adapter():
    print("=" * 60)
    print("TESTING MONITOR ADAPTER")
    print("=" * 60)
    mt = MonitorAdapterTester()
    events = await mt.listen_to_telemetry(duration=20)
    ok = mt.verify_event_structure(events)
    print("\nMONITOR TEST SUMMARY")
    print(f"Events received: {len(events)}")
    print(f"Structure valid: {ok}")
    if events:
        names = sorted({e.get("name", "<unknown>") for e in events})
        print(f"Unique metric names: {len(names)} -> {', '.join(names)}")


async def test_execution_adapter():
    print("=" * 60)
    print("TESTING EXECUTION ADAPTER")
    print("=" * 60)
    et = ExecutionAdapterTester(result_timeout=20.0)
    actions = et.create_test_actions()
    res = await et.test_action_execution(actions)

    print("\nEXECUTION TEST SUMMARY")
    print(f"Actions sent: {len(res['sent_actions'])}")
    print(f"Results received: {len(res['results'])}")
    if res["result_by_action"]:
        succ = sum(1 for r in res["result_by_action"].values() if r.get("success"))
        total = len(res["result_by_action"])
        print(f"Successful actions (by action_id): {succ}/{total}")
    if res["pending_action_ids"]:
        print("Pending action_ids (no result):", ", ".join(res["pending_action_ids"]))
    if res["metrics"]:
        metrics_seen = sorted({m.get("metric", "<unknown>") for m in res["metrics"]})
        print(f"Execution metrics observed ({len(metrics_seen)}): {', '.join(metrics_seen)}")


async def test_both():
    await test_monitor_adapter()
    print("\n")
    await test_execution_adapter()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "monitor":
            asyncio.run(test_monitor_adapter())
        elif cmd == "execution":
            asyncio.run(test_execution_adapter())
        elif cmd == "both":
            asyncio.run(test_both())
        else:
            print("Usage: python verify_adapters.py [monitor|execution|both]")
    else:
        asyncio.run(test_both())
