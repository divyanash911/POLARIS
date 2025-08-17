"""
Unit tests for Enhanced Knowledge Base Storage Implementation.

Tests the enhanced in-memory storage backend for the POLARIS Knowledge Base,
including telemetry optimization, advanced querying, and comprehensive indexing.
"""

import pytest
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from typing import Dict, Any
import uuid

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.kb_storage import InMemoryKBStorage
from polaris.common.query_models import (
    KBEntry,
    KBQuery,
    QueryType,
    KBDataType,
    TelemetryQueryBuilder,
)


class TestEnhancedKBStorageBasics:
    """Test basic functionality of enhanced InMemoryKBStorage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.storage = InMemoryKBStorage(logger=self.logger)

    def test_storage_initialization(self):
        """Test storage initializes correctly with all indexes."""
        assert self.storage.logger is not None
        assert self.storage._entries == {}
        assert self.storage._tag_index == {}
        assert self.storage._metric_index == {}
        assert self.storage._source_index == {}
        assert self.storage._data_type_index == {}
        assert self.storage._label_index == {}
        assert self.storage._time_index == []

    def test_storage_initialization_without_logger(self):
        """Test storage initializes with default logger."""
        storage = InMemoryKBStorage()
        assert storage.logger is not None
        assert hasattr(storage.logger, "debug")
        assert hasattr(storage.logger, "error")

    def test_empty_storage_stats(self):
        """Test comprehensive statistics for empty storage."""
        stats = self.storage.get_stats()

        assert stats["total_entries"] == 0
        assert stats["total_tags"] == 0
        assert stats["total_metrics"] == 0
        assert stats["total_sources"] == 0
        assert stats["total_labels"] == 0
        assert stats["time_indexed_entries"] == 0
        assert "memory_usage_estimate_kb" in stats
        assert stats["memory_usage_estimate_kb"] >= 0
        assert "indexes" in stats
        assert "data_type_breakdown" in stats

    def test_get_data_type_counts(self):
        """Test data type count functionality."""
        counts = self.storage.get_data_type_counts()
        assert isinstance(counts, dict)
        assert len(counts) == 0  # Empty storage


class TestTelemetryDataStorage:
    """Test telemetry-specific storage functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()

        # Create sample telemetry event
        self.telemetry_event = {
            "name": "cpu.usage",
            "value": 85.5,
            "timestamp": "2025-08-17T10:00:00Z",
            "source": "server-001",
            "tags": {"environment": "production", "service": "web"},
            "unit": "percent",
        }

    def test_store_telemetry_event_direct(self):
        """Test storing telemetry event using convenience method."""
        result = self.storage.store_telemetry_event(self.telemetry_event)

        assert result is True

        # Verify entry was created and indexed
        stats = self.storage.get_stats()
        assert stats["total_entries"] == 1
        assert stats["total_metrics"] == 1
        assert stats["total_sources"] == 1
        assert stats["data_type_breakdown"][KBDataType.TELEMETRY_EVENT.value] == 1

    def test_store_telemetry_batch(self):
        """Test storing multiple telemetry events in batch."""
        batch = {
            "events": [
                self.telemetry_event,
                {
                    "name": "memory.usage",
                    "value": 72.3,
                    "timestamp": "2025-08-17T10:01:00Z",
                    "source": "server-001",
                    "tags": {"environment": "production"},
                },
                {
                    "name": "disk.usage",
                    "value": 45.8,
                    "timestamp": "2025-08-17T10:02:00Z",
                    "source": "server-002",
                },
            ]
        }

        result = self.storage.store_telemetry_batch(batch)

        assert result["stored"] == 3
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

        # Verify all entries stored
        stats = self.storage.get_stats()
        assert stats["total_entries"] == 3
        assert stats["total_metrics"] == 3
        assert stats["total_sources"] == 2

    def test_telemetry_entry_field_extraction(self):
        """Test automatic field extraction from telemetry events."""
        entry = KBEntry(
            entry_id="test-telemetry",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=self.telemetry_event,
        )

        self.storage.store(entry)

        # Retrieve and verify extracted fields
        stored_entry = self.storage.get("test-telemetry")
        assert stored_entry.metric_name == "cpu.usage"
        assert stored_entry.metric_value == 85.5
        assert stored_entry.source == "server-001"
        assert stored_entry.event_timestamp == "2025-08-17T10:00:00Z"
        assert stored_entry.labels == {"environment": "production", "service": "web"}

    def test_metric_name_normalization(self):
        """Test metric name normalization."""
        event = {
            "name": "CPU_Usage_Percent",
            "value": 80.0,
            "timestamp": "2025-08-17T10:00:00Z",
        }

        entry = KBEntry(
            entry_id="normalize-test",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=event,
        )

        self.storage.store(entry)

        stored_entry = self.storage.get("normalize-test")
        assert stored_entry.metric_name == "cpu.usage.percent"


class TestEnhancedQuerying:
    """Test enhanced querying capabilities."""

    def setup_method(self):
        """Set up test fixtures with sample data."""
        self.storage = InMemoryKBStorage()

        # Create diverse sample data
        self.sample_entries = [
            # Telemetry events
            KBEntry(
                entry_id="cpu-001",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 85.5,
                    "timestamp": "2025-08-17T10:00:00Z",
                    "source": "server-001",
                    "tags": {"environment": "production", "service": "web"},
                },
            ),
            KBEntry(
                entry_id="memory-001",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "memory.usage",
                    "value": 72.3,
                    "timestamp": "2025-08-17T10:01:00Z",
                    "source": "server-001",
                    "tags": {"environment": "production", "service": "web"},
                },
            ),
            KBEntry(
                entry_id="cpu-002",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 45.2,
                    "timestamp": "2025-08-17T10:02:00Z",
                    "source": "server-002",
                    "tags": {"environment": "staging", "service": "api"},
                },
            ),
            # Alert data
            KBEntry(
                entry_id="alert-001",
                data_type=KBDataType.ALERT,
                content={
                    "title": "High CPU Usage Alert",
                    "severity": "critical",
                    "description": "CPU usage exceeded 80%",
                },
                tags=["alert", "cpu", "critical"],
            ),
        ]

        # Store all entries
        for entry in self.sample_entries:
            self.storage.store(entry)

    def test_structured_query_with_filters(self):
        """Test structured queries with various filters."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 3
        assert all(r.data_type == KBDataType.TELEMETRY_EVENT for r in response.results)

    def test_metric_range_query(self):
        """Test metric range queries."""
        query = TelemetryQueryBuilder.metric_range_query(
            metric_name="cpu.usage", min_value=50.0, max_value=90.0
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].entry_id == "cpu-001"
        assert response.results[0].metric_value == 85.5

    def test_time_series_query(self):
        """Test time series queries."""
        query = TelemetryQueryBuilder.time_series_query(
            metric_name="cpu.usage",
            start_time="2025-08-17T09:00:00Z",
            end_time="2025-08-17T11:00:00Z",
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 2

        # Results should be sorted by timestamp
        timestamps = [r.event_timestamp for r in response.results]
        assert timestamps == sorted(timestamps)

    def test_natural_language_query(self):
        """Test natural language queries."""
        query = TelemetryQueryBuilder.natural_language_query("cpu usage")

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results >= 2  # Should find CPU-related entries

    def test_query_with_sources_filter(self):
        """Test querying with source filtering."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            sources=["server-001"],
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 2
        assert all(r.source == "server-001" for r in response.results)

    def test_query_with_pagination(self):
        """Test query pagination."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
            limit=2,
            offset=1,
        )

        response = self.storage.query(query)

        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 3  # Total available

    def test_advanced_filter_operators(self):
        """Test advanced filter operators."""
        # Test greater than
        query = KBQuery(
            query_type=QueryType.STRUCTURED, filters={"metric_value__gt": 70.0}
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 2
        assert all(r.metric_value > 70.0 for r in response.results)

        # Test contains operator
        query = KBQuery(
            query_type=QueryType.STRUCTURED, filters={"metric_name__contains": "cpu"}
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 2


class TestTelemetrySpecificFeatures:
    """Test telemetry-specific features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()

        # Create time-series data
        base_time = datetime(2025, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            timestamp = base_time.replace(minute=i)

            entry = KBEntry(
                entry_id=f"metric-{i:03d}",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 50.0 + (i * 5),  # Values from 50 to 95
                    "timestamp": timestamp.isoformat(),
                    "source": f"server-{i % 3 + 1:03d}",
                    "tags": {"environment": "production"},
                },
            )
            self.storage.store(entry)

    def test_get_telemetry_metrics(self):
        """Test getting telemetry metrics with filtering."""
        # Get all CPU metrics
        metrics = self.storage.get_telemetry_metrics(metric_name="cpu.usage")

        assert len(metrics) == 10
        assert all(m.metric_name == "cpu.usage" for m in metrics)

        # Get metrics with time range - fix expected count
        metrics = self.storage.get_telemetry_metrics(
            metric_name="cpu.usage",
            start_time="2025-08-17T10:02:00Z",
            end_time="2025-08-17T10:05:00Z",
        )

        assert len(metrics) == 3  # Minutes 2, 3, 4 (end_time is exclusive in range)

    def test_get_metric_summary(self):
        """Test metric summary statistics."""
        summary = self.storage.get_metric_summary("cpu.usage")

        assert summary["metric_name"] == "cpu.usage"
        assert summary["total_entries"] == 10
        assert len(summary["unique_sources"]) == 3
        assert summary["value_statistics"]["min"] == 50.0
        assert summary["value_statistics"]["max"] == 95.0
        assert summary["value_statistics"]["count"] == 10

        # Test non-existent metric - should return dict with metric info, not error
        summary = self.storage.get_metric_summary("non.existent")
        assert summary["metric_name"] == "non.existent"

    def test_get_all_metrics(self):
        """Test getting all unique metrics."""
        metrics = self.storage.get_all_metrics()
        assert "cpu.usage" in metrics
        assert len(metrics) == 1

    def test_get_all_sources(self):
        """Test getting all unique sources."""
        sources = self.storage.get_all_sources()
        assert len(sources) == 3
        assert "server-001" in sources
        assert "server-002" in sources
        assert "server-003" in sources


class TestIndexManagement:
    """Test index management and consistency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()

    def test_tag_index_management(self):
        """Test tag index creation and cleanup."""
        entry = KBEntry(
            entry_id="tag-test",
            data_type=KBDataType.GENERIC,
            content={"data": "test"},
            tags=["tag1", "tag2"],
        )

        self.storage.store(entry)

        # Verify tags indexed
        assert "tag1" in self.storage._tag_index
        assert "tag2" in self.storage._tag_index
        assert "tag-test" in self.storage._tag_index["tag1"]

        # Update tags
        self.storage.update("tag-test", {"tags": ["tag3", "tag4"]})

        # Verify old tags removed, new tags added
        assert "tag1" not in self.storage._tag_index
        assert "tag2" not in self.storage._tag_index
        assert "tag3" in self.storage._tag_index
        assert "tag4" in self.storage._tag_index

    def test_metric_index_management(self):
        """Test metric index management."""
        entry = KBEntry(
            entry_id="metric-test",
            data_type=KBDataType.TELEMETRY_EVENT,
            content={
                "name": "test.metric",
                "value": 100,
                "timestamp": "2025-08-17T10:00:00Z",
            },
        )

        self.storage.store(entry)

        # Verify metric indexed
        assert "test.metric" in self.storage._metric_index
        assert "metric-test" in self.storage._metric_index["test.metric"]

        # Delete entry
        self.storage.delete("metric-test")

        # Verify index cleaned up
        assert "test.metric" not in self.storage._metric_index

    def test_time_index_management(self):
        """Test time index management and ordering."""
        # Create entries with different timestamps
        timestamps = [
            "2025-08-17T10:02:00Z",
            "2025-08-17T10:00:00Z",  # Earlier time
            "2025-08-17T10:01:00Z",
        ]

        for i, timestamp in enumerate(timestamps):
            entry = KBEntry(
                entry_id=f"time-{i}",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={"name": "test.metric", "value": i, "timestamp": timestamp},
            )
            self.storage.store(entry)

        # Verify time index is sorted
        time_entries = [(t, entry_id) for t, entry_id in self.storage._time_index]
        assert len(time_entries) == 3

        # Check sorting (should be in chronological order)
        timestamps_in_index = [t for t, _ in time_entries]
        assert timestamps_in_index == sorted(timestamps_in_index)

    def test_index_effectiveness(self):
        """Test that indexes improve query performance."""
        # Create entries with different characteristics
        for i in range(100):
            entry = KBEntry(
                entry_id=f"perf-{i}",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": f"metric.{i % 10}",
                    "value": float(i),
                    "timestamp": f"2025-08-17T10:{i:02d}:00Z",
                    "source": f"server-{i % 5}",
                },
                tags=[f"tag{i % 3}"],
            )
            self.storage.store(entry)

        # Verify indexes are populated
        stats = self.storage.get_stats()
        assert stats["indexes"]["metrics"] == 10  # 10 unique metrics
        assert stats["indexes"]["sources"] == 5  # 5 unique sources
        assert stats["indexes"]["tags"] == 3  # 3 unique tags

        # Test targeted queries that should use indexes - fix to use valid query
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            sources=["server-0"],
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
        )
        response = self.storage.query(query)
        assert response.total_results == 20  # Every 5th entry


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.storage = InMemoryKBStorage(logger=self.logger)

    def test_storage_exception_handling(self):
        """Test handling of storage exceptions."""
        # Create a problematic entry
        entry = KBEntry(
            entry_id="problem-entry",
            data_type=KBDataType.GENERIC,
            content={"data": "test"},
        )

        # Mock datetime to cause exception
        with patch("polaris.common.kb_storage.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")

            result = self.storage.store(entry)
            assert result is False
            self.logger.error.assert_called()

    def test_query_with_nonexistent_data(self):
        """Test queries on empty storage or non-matching data."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED, filters={"nonexistent": "value"}
        )

        response = self.storage.query(query)

        assert response.success is True
        assert response.total_results == 0
        assert len(response.results) == 0

    def test_malformed_telemetry_event(self):
        """Test handling of malformed telemetry events."""
        malformed_event = {
            "invalid_field": "value"
            # Missing required fields like name, value, timestamp
        }

        # Should still store but with minimal extraction
        result = self.storage.store_telemetry_event(malformed_event)
        assert result is True

        stats = self.storage.get_stats()
        assert stats["total_entries"] == 1

    def test_update_nonexistent_entry(self):
        """Test updating non-existent entries."""
        result = self.storage.update("nonexistent", {"content": {"new": "data"}})
        assert result is False

    def test_delete_nonexistent_entry(self):
        """Test deleting non-existent entries."""
        result = self.storage.delete("nonexistent")
        assert result is False


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()

    def test_bulk_operations_performance(self):
        """Test performance with bulk operations."""
        import time

        # Create large batch of telemetry events
        events = []
        for i in range(1000):
            events.append(
                {
                    "name": f"metric.{i % 10}",
                    "value": float(i % 100),
                    "timestamp": f"2025-08-17T{i//100:02d}:{i%100:02d}:00Z",
                    "source": f"server-{i % 5:03d}",
                }
            )

        batch = {"events": events}

        start_time = time.time()
        result = self.storage.store_telemetry_batch(batch)
        end_time = time.time()

        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0
        assert result["stored"] == 1000
        assert result["failed"] == 0

        # Test query performance
        start_time = time.time()
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
        )
        response = self.storage.query(query)
        end_time = time.time()

        # Query should be fast (less than 1 second)
        assert (end_time - start_time) < 1.0
        assert response.total_results == 1000

    def test_memory_usage_estimation(self):
        """Test memory usage estimation accuracy."""
        initial_stats = self.storage.get_stats()
        initial_memory = initial_stats["memory_usage_estimate_kb"]

        # Add some data
        for i in range(50):
            entry = KBEntry(
                entry_id=f"memory-test-{i}",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "test.metric",
                    "value": float(i),
                    "data": "x" * 1000,  # 1KB of data per entry
                },
            )
            self.storage.store(entry)

        final_stats = self.storage.get_stats()
        final_memory = final_stats["memory_usage_estimate_kb"]

        # Memory usage should have increased significantly
        assert final_memory > initial_memory
        # Should be roughly proportional to data added
        assert final_memory > initial_memory + 40  # At least 40KB increase


class TestKBStorageAdvancedMetricSearch:
    """Test advanced metric search capabilities."""

    def setup_method(self):
        """Set up test fixtures with metric data."""
        self.storage = InMemoryKBStorage()

        # Create entries with various metrics
        self.metric_entries = [
            KBEntry(
                entry_id="server-001",
                content={
                    "hostname": "web-server-001",
                    "cpu_usage": 85.5,
                    "memory_usage": 76.2,
                    "disk_usage": 45,
                    "status": "warning",
                    "metrics": {
                        "cpu": {"usage": 85.5, "cores": 8},
                        "memory": {"total": 16384, "used": 12288},
                        "network": {"bytes_in": 1024000, "bytes_out": 512000},
                    },
                },
                tags=["server", "web", "production"],
                metadata={"datacenter": "us-east-1", "tier": "frontend"},
            ),
            KBEntry(
                entry_id="server-002",
                content={
                    "hostname": "db-server-002",
                    "cpu_usage": 42.1,
                    "memory_usage": 89.7,
                    "disk_usage": 78,
                    "status": "critical",
                    "metrics": {
                        "cpu": {"usage": 42.1, "cores": 16},
                        "memory": {"total": 32768, "used": 29384},
                        "network": {"bytes_in": 2048000, "bytes_out": 1024000},
                    },
                },
                tags=["server", "database", "production"],
                metadata={"datacenter": "us-west-2", "tier": "backend"},
            ),
            KBEntry(
                entry_id="server-003",
                content={
                    "hostname": "cache-server-003",
                    "cpu_usage": 15.3,
                    "memory_usage": 23.8,
                    "disk_usage": 12,
                    "status": "ok",
                    "metrics": {
                        "cpu": {"usage": 15.3, "cores": 4},
                        "memory": {"total": 8192, "used": 1948},
                        "network": {"bytes_in": 512000, "bytes_out": 256000},
                    },
                },
                tags=["server", "cache", "production"],
                metadata={"datacenter": "us-east-1", "tier": "cache"},
            ),
        ]

        for entry in self.metric_entries:
            self.storage.store(entry)

    def test_exact_metric_search(self):
        """Test exact metric value searches."""
        # Exact CPU usage
        results = self.storage.search_by_filters({"cpu_usage": 85.5})
        assert len(results) == 1
        assert results[0].entry_id == "server-001"

        # Exact status
        results = self.storage.search_by_filters({"status": "critical"})
        assert len(results) == 1
        assert results[0].entry_id == "server-002"

    def test_range_metric_searches(self):
        """Test range-based metric searches."""
        # CPU usage greater than 80
        results = self.storage.search_by_filters({"cpu_usage__gt": 80})
        assert len(results) == 1
        assert results[0].entry_id == "server-001"

        # CPU usage greater than or equal to 42.1
        results = self.storage.search_by_filters({"cpu_usage__gte": 42.1})
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

        # Memory usage less than 80
        results = self.storage.search_by_filters({"memory_usage__lt": 80})
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-003" in result_ids

        # Disk usage between 40 and 80
        results = self.storage.search_by_filters(
            {"disk_usage__gte": 40, "disk_usage__lte": 80}
        )
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

    def test_nested_metric_searches(self):
        """Test searches on nested metric fields."""
        # CPU cores greater than 8
        results = self.storage.search_by_filters({"metrics.cpu.cores__gt": 8})
        assert len(results) == 1
        assert results[0].entry_id == "server-002"

        # Memory total exactly 16384
        results = self.storage.search_by_filters({"metrics.memory.total": 16384})
        assert len(results) == 1
        assert results[0].entry_id == "server-001"

        # Network bytes_in greater than 1MB
        results = self.storage.search_by_filters(
            {"metrics.network.bytes_in__gt": 1000000}
        )
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

    def test_string_metric_operations(self):
        """Test string-based metric searches."""
        # Hostname contains "web"
        results = self.storage.search_by_filters({"hostname__contains": "web"})
        assert len(results) == 1
        assert results[0].entry_id == "server-001"

        # Hostname starts with "db"
        results = self.storage.search_by_filters({"hostname__startswith": "db"})
        assert len(results) == 1
        assert results[0].entry_id == "server-002"

        # Status not equal to "ok"
        results = self.storage.search_by_filters({"status__ne": "ok"})
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

    def test_list_operations(self):
        """Test list-based filter operations."""
        # Status in list
        results = self.storage.search_by_filters(
            {"status__in": ["warning", "critical"]}
        )
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

        # Tier not in list
        results = self.storage.search_by_filters(
            {"tier__not_in": ["frontend", "backend"]}
        )
        assert len(results) == 1
        assert results[0].entry_id == "server-003"

    def test_metric_range_convenience_method(self):
        """Test the convenience method for metric ranges."""
        # CPU usage between 40 and 90
        results = self.storage.search_by_metric_range(
            "cpu_usage", min_value=40, max_value=90
        )
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

        # Only minimum value
        results = self.storage.search_by_metric_range("memory_usage", min_value=50)
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-002" in result_ids

        # Only maximum value
        results = self.storage.search_by_metric_range("disk_usage", max_value=50)
        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "server-001" in result_ids
        assert "server-003" in result_ids

    def test_combined_metric_filters(self):
        """Test complex queries with multiple metric filters."""
        # High CPU and high memory
        results = self.storage.search_by_filters(
            {"cpu_usage__gt": 80, "memory_usage__gt": 70}
        )
        assert len(results) == 1
        assert results[0].entry_id == "server-001"

        # Production servers with low resource usage
        results = self.storage.search_by_filters(
            {"tags": ["production"], "cpu_usage__lt": 50, "memory_usage__lt": 50}
        )
        assert len(results) == 1
        assert results[0].entry_id == "server-003"

        # Servers in specific datacenter with very high network traffic (> 1MB)
        results = self.storage.search_by_filters(
            {
                "datacenter": "us-east-1",
                "metrics.network.bytes_in__gt": 1000000,  # Changed from 500000 to 1000000
            }
        )
        assert len(results) == 1
        assert results[0].entry_id == "server-001"

    def test_invalid_operators(self):
        """Test handling of invalid operators."""
        # Invalid operator should not match anything
        results = self.storage.search_by_filters({"cpu_usage__invalid": 50})
        assert len(results) == 0

        # Invalid field should not match anything
        results = self.storage.search_by_filters({"nonexistent_field": "value"})
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
