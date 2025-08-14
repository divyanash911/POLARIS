"""
SWIM Execution Adapter for POLARIS POC
"""

import asyncio
import json
import logging
from pathlib import Path
import signal
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg


from polaris.common import setup_logging, now_iso, jittered_backoff
from polaris.common.config import load_config, get_config


load_config(search_paths=[Path("/home/prakhar/dev/prakhar479/POLARIS/polaris_poc/src/config/polaris_config.yaml")],
            required_keys=["SWIM_HOST", "SWIM_PORT", "NATS_URL"])


# --------------------------------- Data Model ---------------------------------

class ActionType(Enum):
    ADD_SERVER = "ADD_SERVER"
    REMOVE_SERVER = "REMOVE_SERVER"
    ADJUST_QOS = "ADJUST_QOS"
    SET_DIMMER = "SET_DIMMER"  # alias for ADJUST_QOS


@dataclass
class ControlAction:
    """Structure for control actions from POLARIS."""
    action_type: str
    timestamp: str
    source: str
    params: Dict[str, Any]
    action_id: str  # for correlation/idempotency

    @classmethod
    def from_json(cls, json_str: str) -> "ControlAction":
        data = json.loads(json_str)
        action_id = data.get("action_id") or str(uuid.uuid4())
        return cls(
            action_type=data.get("action_type", ""),
            timestamp=data.get("timestamp", now_iso()),
            source=data.get("source", "unknown"),
            params=data.get("params", {}) or {},
            action_id=action_id,
        )


@dataclass
class ExecutionResult:
    """Result of action execution."""
    action_id: str
    action_type: str
    success: bool
    message: str
    started_at: str
    finished_at: str
    duration_sec: float

    def to_json(self) -> bytes:
        return json.dumps({
            "action_id": self.action_id,
            "action_type": self.action_type,
            "success": self.success,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": round(self.duration_sec, 6),
        }).encode()


# ---------------------------- SWIM Async TCP Client ----------------------------

class SwimExecutionClientAsync:
    """
    Async TCP client for SWIM External Control mode.

    We connect fresh per command (prevents 'stuck' sockets while SWIM is busy).
    Protocol: simple line-based commands; response expected as one line.
    """

    def __init__(self, host: str, port: int, timeout_sec: float, logger: logging.Logger):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.logger = logger

    async def _send_recv(self, command: str) -> str:
        """Open connection, send single command with newline, read single line response."""
        start = time.perf_counter()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout_sec,
            )
        except Exception as e:
            raise TimeoutError(f"connect failed: {e}")

        try:
            # Send
            writer.write((command + "\n").encode())
            await asyncio.wait_for(writer.drain(), timeout=self.timeout_sec)

            # Receive a line
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout_sec)
            resp = line.decode(errors="replace").strip()
            return resp
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"command timeout: {e}")
        except Exception as e:
            raise RuntimeError(f"command error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            self.logger.debug("swim_tcp", extra={
                "phase": "io_complete",
                "cmd": command,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
            })

    async def send_with_retries(self, command: str, max_retries: int,
                                base_delay: float, max_delay: float,
                                action_ctx: Dict[str, Any]) -> str:
        """
        Send a command with retry/backoff on timeouts/transient errors.
        """
        attempt = 0
        while True:
            try:
                self.logger.info("swim_send", extra={
                                 **action_ctx, "cmd": command, "attempt": attempt})
                resp = await self._send_recv(command)
                self.logger.info("swim_resp", extra={
                                 **action_ctx, "cmd": command, "resp": resp, "attempt": attempt})
                return resp
            except (TimeoutError, ConnectionError, RuntimeError, OSError) as e:
                if attempt >= max_retries:
                    self.logger.error("swim_failed", extra={
                                      **action_ctx, "cmd": command, "attempt": attempt, "error": str(e)})
                    raise
                delay = jittered_backoff(attempt, base_delay, max_delay)
                self.logger.warning("swim_retry", extra={
                                    **action_ctx, "cmd": command, "attempt": attempt, "error": str(e), "retry_in_sec": round(delay, 3)})
                await asyncio.sleep(delay)
                attempt += 1


# ------------------------------- Execution Adapter (NATS) --------------------------------

class SwimExecutionAdapter:
    def __init__(self):
        # Config
        self.nats_url = get_config("NATS_URL", "nats://localhost:4222")
        self.swim_host = get_config("SWIM_HOST", "localhost")
        self.swim_port = get_config("SWIM_PORT", "4242", int)
        self.swim_cmd_timeout = get_config("SWIM_CMD_TIMEOUT", "30", float)
        self.swim_max_retries = get_config("SWIM_MAX_RETRIES", "2", int)
        self.retry_base_delay = get_config(
            "SWIM_RETRY_BASE_DELAY", "1.0", float)
        self.retry_max_delay = get_config("SWIM_RETRY_MAX_DELAY", "5.0", float)
        self.min_gap_between_actions = get_config(
            "SWIM_MIN_GAP_BETWEEN_ACTIONS", "0", float)
        self.queue_maxsize = get_config("EXECUTION_QUEUE_MAXSIZE", "1000", int)
        self.action_subject = get_config("EXECUTION_ACTION_SUBJECT", "polaris.execution.actions")

        # Logging
        self.logger = setup_logging()
        self.logger.info("adapter_init", extra={
            "nats_url": self.nats_url,
            "swim_host": self.swim_host,
            "swim_port": self.swim_port,
            "swim_cmd_timeout": self.swim_cmd_timeout,
            "swim_max_retries": self.swim_max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "min_gap_between_actions": self.min_gap_between_actions,
            "queue_maxsize": self.queue_maxsize,
        })

        # NATS
        self.nc: Optional[NATS] = None

        # Queue & worker
        self.queue: asyncio.Queue[ControlAction] = asyncio.Queue(
            maxsize=self.queue_maxsize if self.queue_maxsize > 0 else 0)
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None

        # SWIM client
        self.swim = SwimExecutionClientAsync(
            host=self.swim_host,
            port=self.swim_port,
            timeout_sec=self.swim_cmd_timeout,
            logger=self.logger,
        )

        # Strike counter (semantic warnings)
        self.strike_count = 0

        # Last action finish timestamp (for min-gap throttling)
        self._last_action_finished_at: float = 0.0

    # -------------------------- Metrics Publishing --------------------------

    async def publish_metric(self, metric_name: str, payload: Dict[str, Any]):
        if not self.nc:
            return
        msg = {
            "metric": metric_name,
            "ts": now_iso(),
            **payload,
        }
        try:
            await self.nc.publish("polaris.execution.metrics", json.dumps(msg).encode())
        except Exception as e:
            self.logger.warning("metric_publish_failed", extra={
                                "metric": metric_name, "error": str(e)})

    # ---------------------------- Action Execution ---------------------------

    async def execute_action(self, action: ControlAction) -> ExecutionResult:
        started_at = now_iso()
        t0 = time.perf_counter()

        ctx = {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "source": action.source,
        }

        try:
            atype = action.action_type

            if atype == ActionType.ADD_SERVER.value:
                result_msg, ok = await self._add_server(ctx)

            elif atype == ActionType.REMOVE_SERVER.value:
                result_msg, ok = await self._remove_server(ctx)

            elif atype in (ActionType.ADJUST_QOS.value, ActionType.SET_DIMMER.value):
                dimmer_value = action.params.get(
                    "value", action.params.get("dimmer_value"))
                if dimmer_value is None:
                    msg = "Missing dimmer value (params.value or params.dimmer_value)"
                    self.logger.warning("invalid_params", extra={
                                        **ctx, "detail": msg})
                    result_msg, ok = msg, False
                else:
                    try:
                        val = float(dimmer_value)
                    except Exception:
                        msg = f"Invalid dimmer value (not a number): {dimmer_value}"
                        self.logger.warning("invalid_params", extra={
                                            **ctx, "detail": msg})
                        result_msg, ok = msg, False
                    else:
                        if not (0.0 <= val <= 1.0):
                            self.strike_count += 1
                            msg = f"Strike {self.strike_count} - Invalid dimmer value: {val} (must be 0.0-1.0)"
                            self.logger.warning("invalid_range", extra={
                                                **ctx, "detail": msg})
                            result_msg, ok = msg, False
                        else:
                            resp = await self._send_swim(f"set_dimmer {val}", ctx)
                            result_msg, ok = resp, not resp.lower().startswith("error")

            else:
                msg = f"Unknown action type '{atype}'"
                self.logger.error("unknown_action", extra={
                                  **ctx, "detail": msg})
                result_msg, ok = msg, False

        except Exception as e:
            result_msg, ok = f"Exception: {e}", False
            self.logger.exception("execute_action_exception", extra=ctx)

        finished_at = now_iso()
        duration_sec = time.perf_counter() - t0

        er = ExecutionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            success=ok,
            message=result_msg,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
        )

        # Publish timing metric
        await self.publish_metric("swim.execution.action_duration", {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "duration_sec": round(duration_sec, 6),
            "success": ok,
        })

        return er

    async def _send_swim(self, command: str, ctx: Dict[str, Any]) -> str:
        """Wrapper to send SWIM command with retries/backoff."""
        return await self.swim.send_with_retries(
            command=command,
            max_retries=self.swim_max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            action_ctx=ctx,
        )

    async def _get_int(self, cmd: str, ctx: Dict[str, Any]) -> Optional[int]:
        try:
            resp = await self._send_swim(cmd, ctx)
            return int(resp)
        except Exception as e:
            self.logger.warning("get_int_failed", extra={
                                **ctx, "cmd": cmd, "error": str(e)})
            return None

    async def _add_server(self, ctx: Dict[str, Any]) -> Tuple[str, bool]:
        curr = await self._get_int("get_servers", ctx)
        maxs = await self._get_int("get_max_servers", ctx)
        if curr is None or maxs is None:
            # Best effort: still try add_server (SWIM may respond even if queries failed)
            self.logger.warning("server_counts_unknown", extra=ctx)
        if curr is not None and maxs is not None and curr >= maxs:
            self.strike_count += 1
            msg = f"Strike {self.strike_count} - Cannot add server: already at maximum ({maxs})"
            self.logger.warning("add_server_blocked", extra={
                                **ctx, "detail": msg, "current": curr, "max": maxs})
            return msg, False

        resp = await self._send_swim("add_server", ctx)
        ok = not resp.lower().startswith("error")
        return resp, ok

    async def _remove_server(self, ctx: Dict[str, Any]) -> Tuple[str, bool]:
        curr = await self._get_int("get_servers", ctx)
        if curr is not None and curr <= 1:
            self.strike_count += 1
            msg = f"Strike {self.strike_count} - Cannot remove server: only {curr} server remaining"
            self.logger.warning("remove_server_blocked", extra={
                                **ctx, "detail": msg, "current": curr})
            return msg, False

        resp = await self._send_swim("remove_server", ctx)
        ok = not resp.lower().startswith("error")
        return resp, ok

    # ------------------------------- NATS Layer --------------------------------

    async def connect_nats(self) -> bool:
        try:
            self.nc = NATS()
            await self.nc.connect(self.nats_url)
            self.logger.info("nats_connected", extra={
                             "nats_url": self.nats_url})
            return True
        except Exception as e:
            self.logger.error("nats_connect_failed", extra={
                              "error": str(e), "nats_url": self.nats_url})
            return False

    async def publish_execution_result(self, result: ExecutionResult):
        if not self.nc:
            return
        try:
            await self.nc.publish("polaris.execution.results", result.to_json())
            self.logger.info("result_published", extra={
                "action_id": result.action_id,
                "action_type": result.action_type,
                "success": result.success,
                "duration_sec": round(result.duration_sec, 6),
            })
        except Exception as e:
            self.logger.error("result_publish_failed", extra={
                "action_id": result.action_id,
                "action_type": result.action_type,
                "error": str(e),
            })

    async def _enqueue(self, action: ControlAction):
        qsize_before = self.queue.qsize()
        await self.queue.put(action)
        qsize_after = self.queue.qsize()
        self.logger.info("action_enqueued", extra={
            "action_id": action.action_id,
            "action_type": action.action_type,
            "source": action.source,
            "queue_size_before": qsize_before,
            "queue_size_after": qsize_after,
        })
        await self.publish_metric("swim.execution.queue_length", {
            "queue_length": qsize_after
        })

    async def action_handler(self, msg: Msg):
        try:
            action = ControlAction.from_json(msg.data.decode())
            await self._enqueue(action)
        except Exception as e:
            self.logger.error("action_parse_failed", extra={"error": str(
                e), "raw": msg.data[:256].decode(errors="replace")})

    async def worker(self):
        self.logger.info("worker_started")
        try:
            while self.running:
                action: ControlAction = await self.queue.get()
                ctx = {"action_id": action.action_id,
                       "action_type": action.action_type}

                # Optional throttle to respect SWIM boot/processing time
                if self.min_gap_between_actions > 0 and self._last_action_finished_at > 0:
                    since_last = time.perf_counter() - self._last_action_finished_at
                    wait_more = self.min_gap_between_actions - since_last
                    if wait_more > 0:
                        self.logger.info("worker_throttle", extra={
                                         **ctx, "sleep_sec": round(wait_more, 3)})
                        await asyncio.sleep(wait_more)

                qsize_before = self.queue.qsize()
                self.logger.info("action_processing_start", extra={
                                 **ctx, "queue_size_before": qsize_before})

                try:
                    result = await self.execute_action(action)
                    await self.publish_execution_result(result)
                except Exception as e:
                    # This is very defensive; execute_action already handles most exceptions
                    self.logger.exception("action_processing_exception", extra={
                                          **ctx, "error": str(e)})

                self._last_action_finished_at = time.perf_counter()
                qsize_after = self.queue.qsize()
                self.logger.info("action_processing_end", extra={
                                 **ctx, "queue_size_after": qsize_after})
                await self.publish_metric("swim.execution.queue_length", {"queue_length": qsize_after})

                self.queue.task_done()
        except asyncio.CancelledError:
            self.logger.info("worker_cancelled")
        finally:
            self.logger.info("worker_stopped")

    async def start(self):
        self.logger.info("adapter_starting")
        ok = await self.connect_nats()
        if not ok:
            self.logger.error("adapter_start_failed", extra={
                              "reason": "nats_connect"})
            return

        # Subscribe
        try:
            assert self.nc is not None
            await self.nc.subscribe(self.action_subject, cb=self.action_handler)
            self.logger.info("nats_subscribed", extra={
                "subjects": [self.action_subject]
            })
        except Exception as e:
            self.logger.error("nats_subscribe_failed", extra={"error": str(e)})
            return

        # Run worker
        self.running = True
        self.worker_task = asyncio.create_task(self.worker())
        self.logger.info("adapter_started", extra={"status": "running"})

    async def drain_and_stop(self, drain_timeout: float = 10.0):
        """Graceful shutdown: stop intake, drain queue for a bit, close NATS."""
        self.logger.info("adapter_stopping")
        self.running = False

        # Stop taking new actions by unsubscribing
        try:
            if self.nc and self.nc._subs:
                # Cancel all subscriptions
                for sid in list(self.nc._subs.keys()):
                    await self.nc.unsubscribe(sid)
        except Exception:
            pass

        # Allow worker to finish current item + some draining time
        try:
            if self.worker_task:
                await asyncio.wait_for(self.queue.join(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("drain_timeout_reached", extra={
                                "remaining": self.queue.qsize()})

        # Cancel worker if still running
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        # Close NATS
        try:
            if self.nc:
                await self.nc.close()
        except Exception:
            pass

        self.logger.info("adapter_stopped")


# ----------------------------------- Main -------------------------------------

async def main():
    adapter = SwimExecutionAdapter()

    # Start adapter
    await adapter.start()

    # Handle signals for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        adapter.logger.info("signal_received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows
            signal.signal(sig, lambda *_: _signal_handler())

    # Run until signaled
    await stop_event.wait()
    await adapter.drain_and_stop(drain_timeout=10.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # As a last resort fallback
        pass
