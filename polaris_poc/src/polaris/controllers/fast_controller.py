# polaris_poc/src/controllers/fast_controller.py

# from polaris.kernel.kernel import SWIMKernel
from polaris.models.telemetry import TelemetryBatch, TelemetryEvent
import json

# Base class
class BaseController:
    def __init__(self, kernel):
        """
        Initialize the controller with a reference to the kernel.
        """
        self.kernel = kernel

    def decide_action(self, telemetry):
        """
        Decide an action based on the telemetry data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

# FastController class
class FastController(BaseController):
    def decide_action(self, telemetry: TelemetryBatch):
        """
        Decide an action based on the telemetry data.
        """
        events = telemetry.get("events", [])
        avg_values = {}
        for event in events:
            name = event.get("name")
            value = event.get("value")
            if name not in avg_values:
                avg_values[name] = []
            avg_values[name].append(value)

        if avg_values["swim.server.utilization"][0] > 0.5:
            return {
                "action_type": "REMOVE_SERVER",
                "source": "fast_controller",
                "action_id": "123e4567-e89b-12d3-a456-426614174000",
                "params": {"server_type": "compute", "count": 1},
                "priority": "normal",
            }
        else:
            return {
                "action_type": "ADD_SERVER",
                "source": "fast_controller",
                "action_id": "123e4567-e89b-12d3-a456-426614174000",
                "params": {"server_type": "compute", "count": 1},
                "priority": "normal",
            }

# TestController class
class TestController(BaseController):
    def decide_action(self, telemetry: TelemetryBatch):
        """
        Simple implementation for testing purposes.
        """
        return {
            "action_type": "TEST_ACTION",
            "source": "test_controller",
            "action_id": "test-1234",
            "params": {"test_param": "value"},
            "priority": "low",
        }
        

        


