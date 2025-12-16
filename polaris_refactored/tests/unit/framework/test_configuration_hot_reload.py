import time
import asyncio
from pathlib import Path
import pytest

from framework.configuration.core import PolarisConfiguration
from framework.configuration.sources import YAMLConfigurationSource


@pytest.mark.asyncio
async def test_reload_callbacks_invoked_on_reload(tmp_path: Path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("framework:\n  logging_config:\n    level: INFO\n", encoding="utf-8")

    cfg = PolarisConfiguration([YAMLConfigurationSource(cfg_file)], enable_hot_reload=False)

    calls = {"n": 0}

    def on_reload():
        calls["n"] += 1

    cfg.add_reload_callback(on_reload)

    # Change file and reload
    cfg_file.write_text("framework:\n  logging_config:\n    level: DEBUG\n", encoding="utf-8")
    # Change file and reload
    cfg_file.write_text("framework:\n  logging_config:\n    level: DEBUG\n", encoding="utf-8")
    cfg.reload_configuration()
    await asyncio.sleep(0.5) # Yield to event loop, increased wait

    assert calls["n"] == 1
    assert cfg.get_framework_config().logging_config.level == "DEBUG"


@pytest.mark.asyncio
async def test_hot_reload_thread_detects_file_change(tmp_path: Path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("framework:\n  logging_config:\n    level: INFO\n", encoding="utf-8")

    cfg = PolarisConfiguration([YAMLConfigurationSource(cfg_file)], enable_hot_reload=True)

    calls = {"n": 0}

    def on_reload():
        calls["n"] += 1

    cfg.add_reload_callback(on_reload)

    # Allow the background thread to initialize and capture the initial mtime
    await asyncio.sleep(0.5)

    # Touch file with new content to trigger change
    cfg_file.write_text("framework:\n  logging_config:\n    level: WARNING\n", encoding="utf-8")

    # Wait up to ~5s for background thread (checks every 1s) to detect
    # Wait up to ~5s for background thread (checks every 1s) to detect
    start = time.time()
    while time.time() - start < 5.0 and calls["n"] < 1:
        await asyncio.sleep(0.2)
    
    cfg.stop_hot_reload()

    assert calls["n"] >= 1
