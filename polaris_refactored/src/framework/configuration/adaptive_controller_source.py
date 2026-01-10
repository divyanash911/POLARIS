"""
Adaptive Controller Configuration Source

Specialized configuration source for the adaptive controller that supports
evolvable thresholds and integration with the meta learner.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timezone

import yaml

from .sources import ConfigurationSource
from infrastructure.exceptions import ConfigurationError


class AdaptiveControllerConfigurationSource(ConfigurationSource):
    """Configuration source for adaptive controller with evolvable thresholds.
    
    This source provides:
    - Hot reload detection for configuration changes
    - Validation of threshold constraints
    - Evolution metadata tracking
    - Integration with meta learner updates
    """
    
    def __init__(
        self, 
        file_path: Path, 
        priority: int = 100,
        enable_evolution: bool = True,
        evolution_callbacks: Optional[List[Callable[[str, float, float], None]]] = None
    ):
        self.file_path = Path(file_path)
        self._priority = priority
        self._last_modified = None
        self._cached_config = None
        self._enable_evolution = enable_evolution
        self._evolution_callbacks = evolution_callbacks or []
        
        # Logger
        self.logger = logging.getLogger("polaris.config.adaptive_controller")
        
        # Evolution tracking
        self._evolution_history: Dict[str, List[Dict[str, Any]]] = {}
        self._last_evolution_check = datetime.now(timezone.utc)
    
    async def load(self) -> Dict[str, Any]:
        """Load adaptive controller configuration with evolvable threshold support."""
        try:
            if not self.file_path.exists():
                raise ConfigurationError(
                    f"Adaptive controller config file not found: {self.file_path}",
                    config_path=str(self.file_path)
                )
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Process evolvable thresholds
            config = self._process_evolvable_thresholds(raw_config)
            
            # Validate configuration
            self._validate_config(config)
            
            # Update cache
            self._cached_config = config
            self._last_modified = self.file_path.stat().st_mtime
            
            # Check for evolution changes
            self._check_evolution_changes(config)
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in adaptive controller config: {self.file_path}",
                config_path=str(self.file_path),
                validation_errors=[str(e)]
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error loading adaptive controller config: {self.file_path}",
                config_path=str(self.file_path),
                validation_errors=[str(e)]
            )
    
    def _process_evolvable_thresholds(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process evolvable thresholds and convert legacy format if needed."""
        thresholds = config.get("thresholds", {})
        processed_thresholds = {}
        
        for name, threshold_data in thresholds.items():
            if isinstance(threshold_data, (int, float)):
                # Legacy format - convert to evolvable threshold
                processed_thresholds[name] = {
                    "value": float(threshold_data),
                    "min_value": self._get_default_min_value(name),
                    "max_value": self._get_default_max_value(name),
                    "metadata": {
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "updated_by": "system",
                        "confidence_score": 1.0,
                        "performance_impact": None,
                        "change_reason": "legacy_conversion",
                        "previous_value": None,
                        "evolution_count": 0
                    }
                }
                self.logger.info(f"Converted legacy threshold {name} to evolvable format")
            elif isinstance(threshold_data, dict):
                # Already in evolvable format
                processed_thresholds[name] = threshold_data
            else:
                self.logger.warning(f"Invalid threshold format for {name}: {type(threshold_data)}")
                continue
        
        # Update config with processed thresholds
        result = config.copy()
        result["thresholds"] = processed_thresholds
        return result
    
    def _get_default_min_value(self, threshold_name: str) -> float:
        """Get default minimum value for a threshold."""
        defaults = {
            "cpu_high": 50.0,
            "cpu_low": 5.0,
            "cpu_critical": 85.0,
            "memory_high": 60.0,
            "memory_critical": 85.0,
            "response_time_warning_ms": 50.0,
            "response_time_critical_ms": 200.0,
            "error_rate_warning_pct": 1.0,
            "error_rate_critical_pct": 5.0,
            "throughput_low": 1.0,
        }
        return defaults.get(threshold_name, 0.0)
    
    def _get_default_max_value(self, threshold_name: str) -> float:
        """Get default maximum value for a threshold."""
        defaults = {
            "cpu_high": 95.0,
            "cpu_low": 50.0,
            "cpu_critical": 99.0,
            "memory_high": 95.0,
            "memory_critical": 99.0,
            "response_time_warning_ms": 1000.0,
            "response_time_critical_ms": 2000.0,
            "error_rate_warning_pct": 20.0,
            "error_rate_critical_pct": 30.0,
            "throughput_low": 100.0,
        }
        return defaults.get(threshold_name, 100.0)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate adaptive controller configuration."""
        # Validate required sections
        required_sections = ["thresholds", "cooldowns", "limits", "features"]
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(
                    f"Missing required section '{section}' in adaptive controller config",
                    config_path=str(self.file_path),
                    validation_errors=[f"Missing section: {section}"]
                )
        
        # Validate thresholds
        thresholds = config["thresholds"]
        for name, threshold_data in thresholds.items():
            if not isinstance(threshold_data, dict):
                continue
            
            value = threshold_data.get("value")
            min_value = threshold_data.get("min_value")
            max_value = threshold_data.get("max_value")
            
            if value is None or min_value is None or max_value is None:
                raise ConfigurationError(
                    f"Threshold {name} missing required fields (value, min_value, max_value)",
                    config_path=str(self.file_path),
                    validation_errors=[f"Invalid threshold: {name}"]
                )
            
            if not (min_value <= value <= max_value):
                raise ConfigurationError(
                    f"Threshold {name} value {value} not within bounds [{min_value}, {max_value}]",
                    config_path=str(self.file_path),
                    validation_errors=[f"Threshold out of bounds: {name}"]
                )
        
        # Validate evolution settings if enabled
        if self._enable_evolution:
            evolution = config.get("evolution", {})
            if evolution.get("enable_threshold_evolution", False):
                confidence_threshold = evolution.get("evolution_confidence_threshold", 0.7)
                if not 0.0 <= confidence_threshold <= 1.0:
                    raise ConfigurationError(
                        f"Evolution confidence threshold {confidence_threshold} must be between 0.0 and 1.0",
                        config_path=str(self.file_path),
                        validation_errors=["Invalid evolution confidence threshold"]
                    )
    
    def _check_evolution_changes(self, config: Dict[str, Any]) -> None:
        """Check for threshold evolution changes and notify callbacks."""
        if not self._enable_evolution or not self._cached_config:
            return
        
        current_thresholds = config.get("thresholds", {})
        previous_thresholds = self._cached_config.get("thresholds", {})
        
        for name, current_data in current_thresholds.items():
            if name not in previous_thresholds:
                continue
            
            current_value = current_data.get("value")
            previous_value = previous_thresholds[name].get("value")
            
            if current_value != previous_value:
                # Record evolution
                evolution_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "previous_value": previous_value,
                    "new_value": current_value,
                    "metadata": current_data.get("metadata", {})
                }
                
                if name not in self._evolution_history:
                    self._evolution_history[name] = []
                self._evolution_history[name].append(evolution_record)
                
                # Notify callbacks
                for callback in self._evolution_callbacks:
                    try:
                        callback(name, previous_value, current_value)
                    except Exception as e:
                        self.logger.error(f"Error in evolution callback: {e}")
                
                self.logger.info(
                    f"Threshold evolution detected: {name} changed from {previous_value} to {current_value}"
                )
    
    def get_priority(self) -> int:
        """Get the priority of this configuration source."""
        return self._priority
    
    def has_changed(self) -> bool:
        """Check if the configuration source has changed."""
        if not self.file_path.exists():
            return False
        
        current_mtime = self.file_path.stat().st_mtime
        return current_mtime != self._last_modified
    
    def get_evolution_history(self, threshold_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get evolution history for thresholds."""
        if threshold_name:
            return {threshold_name: self._evolution_history.get(threshold_name, [])}
        return self._evolution_history.copy()
    
    def add_evolution_callback(self, callback: Callable[[str, float, float], None]) -> None:
        """Add a callback to be notified of threshold evolution changes."""
        self._evolution_callbacks.append(callback)
    
    async def update_threshold(
        self, 
        threshold_name: str, 
        new_value: float,
        updated_by: str = "meta_learner",
        reason: str = "optimization",
        confidence: float = 1.0,
        performance_impact: Optional[float] = None
    ) -> bool:
        """Update a threshold value and save to configuration file."""
        try:
            # Load current config
            config = await self.load()
            thresholds = config.get("thresholds", {})
            
            if threshold_name not in thresholds:
                self.logger.error(f"Threshold {threshold_name} not found in configuration")
                return False
            
            threshold_data = thresholds[threshold_name]
            min_value = threshold_data.get("min_value", 0.0)
            max_value = threshold_data.get("max_value", 100.0)
            
            # Validate new value
            if not (min_value <= new_value <= max_value):
                self.logger.error(
                    f"New value {new_value} for {threshold_name} not within bounds [{min_value}, {max_value}]"
                )
                return False
            
            # Update threshold
            previous_value = threshold_data.get("value")
            threshold_data["value"] = new_value
            
            # Update metadata
            metadata = threshold_data.setdefault("metadata", {})
            metadata["previous_value"] = previous_value
            metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
            metadata["updated_by"] = updated_by
            metadata["confidence_score"] = confidence
            metadata["performance_impact"] = performance_impact
            metadata["change_reason"] = reason
            metadata["evolution_count"] = metadata.get("evolution_count", 0) + 1
            
            # Update global metadata
            config["last_updated"] = datetime.now(timezone.utc).isoformat()
            config["updated_by"] = updated_by
            
            # Save to file
            with open(self.file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(
                f"Updated threshold {threshold_name} from {previous_value} to {new_value} "
                f"(reason: {reason}, confidence: {confidence:.2f})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update threshold {threshold_name}: {e}")
            return False