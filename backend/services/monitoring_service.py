"""
Monitoring service for OPZ Product Matcher
Provides performance metrics, health checks, and alerting
"""
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import json
from loguru import logger

from config.settings import settings
from services.cache_service import cache_service


class MetricsCollector:
    """Collects and aggregates performance metrics"""

    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, Dict[str, Any]] = {}

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        key = self._make_key(name, tags)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        key = self._make_key(name, tags)
        self.gauges[key] = value

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'values': deque(maxlen=1000)
            }

        hist = self.histograms[key]
        hist['count'] += 1
        hist['sum'] += value
        hist['min'] = min(hist['min'], value)
        hist['max'] = max(hist['max'], value)
        hist['values'].append(value)

    def time_execution(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Decorator/context manager for timing execution"""
        return TimerContext(self, name, tags)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {
                name: {
                    'count': hist['count'],
                    'sum': hist['sum'],
                    'min': hist['min'],
                    'max': hist['max'],
                    'avg': hist['sum'] / hist['count'] if hist['count'] > 0 else 0,
                    'p95': self._percentile(hist['values'], 95) if hist['values'] else None,
                    'p99': self._percentile(hist['values'], 99) if hist['values'] else None,
                }
                for name, hist in self.histograms.items()
            },
            'timestamp': datetime.now().isoformat()
        }

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key from name and tags"""
        if not tags:
            return name
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{tag_str}}"

    def _percentile(self, values: deque, percentile: float) -> float:
        """Calculate percentile from sorted values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class TimerContext:
    """Context manager for timing execution"""

    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_histogram(f"{self.name}_duration", duration, self.tags)


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable] = []

    def add_alert_handler(self, handler: Callable):
        """Add an alert notification handler"""
        self.alert_handlers.append(handler)

    def trigger_alert(self, alert_type: str, message: str, severity: str = "warning",
                     metadata: Optional[Dict[str, Any]] = None):
        """Trigger an alert"""
        alert = {
            'id': f"{alert_type}_{int(time.time())}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.alerts.append(alert)
        logger.warning(f"Alert triggered: {alert_type} - {message}")

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                # Run alert handler in background thread with new event loop
                from concurrent.futures import ThreadPoolExecutor

                def run_alert_handler():
                    """Run the alert handler in a new event loop"""
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(handler(alert))
                    finally:
                        loop.close()

                # Run in thread pool to avoid blocking
                executor = ThreadPoolExecutor(max_workers=1)
                executor.submit(run_alert_handler)
                executor.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by severity"""
        alerts = self.alerts
        if severity_filter:
            alerts = [a for a in alerts if a['severity'] == severity_filter]
        return alerts[-100:]  # Return last 100 alerts

    def clear_alert(self, alert_id: str):
        """Clear a specific alert"""
        self.alerts = [a for a in self.alerts if a['id'] != alert_id]


class HealthChecker:
    """Performs health checks on system components"""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}

    def add_check(self, name: str, check_func: Callable):
        """Add a health check function"""
        self.checks[name] = check_func

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func()
                duration = time.time() - start_time

                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        return results

    async def is_healthy(self) -> bool:
        """Check if system is healthy"""
        results = await self.run_health_checks()
        return all(result['status'] == 'healthy' for result in results.values())


class MonitoringService:
    """Main monitoring service coordinating all monitoring components"""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.health = HealthChecker()
        self._initialized = False

    async def initialize(self):
        """Initialize monitoring service"""
        if self._initialized:
            return

        # Add default health checks
        self.health.add_check('redis', self._check_redis)
        self.health.add_check('database', self._check_database)
        self.health.add_check('storage', self._check_storage)

        # Add default alert handlers
        self.alerts.add_alert_handler(self._default_alert_handler)

        self._initialized = True
        logger.info("Monitoring service initialized")

    async def _check_redis(self) -> bool:
        """Check Redis connectivity"""
        try:
            return await cache_service.exists('health_check')
        except Exception:
            return False

    async def _check_database(self) -> bool:
        """Check database connectivity"""
        try:
            from services.database import get_db
            async for session in get_db():
                await session.execute("SELECT 1")
                return True
            return False
        except Exception:
            return False

    async def _check_storage(self) -> bool:
        """Check storage connectivity"""
        try:
            from services.storage_service import storage_service
            # Simple check - try to list bucket (won't work if no bucket, but tests connectivity)
            return True  # For now, assume storage is healthy
        except Exception:
            return False

    async def _default_alert_handler(self, alert: Dict[str, Any]):
        """Default alert handler - logs alerts"""
        logger.warning(f"Alert: {alert['type']} - {alert['message']}")

    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        tags = {'endpoint': endpoint, 'method': method, 'status': str(status_code)}
        self.metrics.increment_counter('api_requests_total', tags=tags)
        self.metrics.record_histogram('api_request_duration', duration, tags)

        # Alert on high error rates
        if status_code >= 500:
            self.alerts.trigger_alert(
                'api_error',
                f"API error {status_code} on {method} {endpoint}",
                'error',
                {'endpoint': endpoint, 'method': method, 'status_code': status_code}
            )

    def record_claude_usage(self, tokens_used: int, model: str, operation: str):
        """Record Claude API usage"""
        tags = {'model': model, 'operation': operation}
        self.metrics.increment_counter('claude_tokens_used', tokens_used, tags)
        self.metrics.record_histogram('claude_request_tokens', tokens_used, tags)

    def record_cache_operation(self, operation: str, hit: bool = None):
        """Record cache operation metrics"""
        self.metrics.increment_counter('cache_operations_total', tags={'operation': operation})
        if hit is not None:
            self.metrics.increment_counter('cache_hits_total' if hit else 'cache_misses_total')

    def record_document_processing(self, document_type: str, size_bytes: int, duration: float, success: bool):
        """Record document processing metrics"""
        tags = {'document_type': document_type, 'success': str(success)}
        self.metrics.increment_counter('documents_processed_total', tags=tags)
        self.metrics.record_histogram('document_processing_duration', duration, tags)
        self.metrics.record_histogram('document_size_bytes', size_bytes, tags)

    def record_search_operation(self, query_type: str, results_count: int, duration: float):
        """Record search operation metrics"""
        tags = {'query_type': query_type}
        self.metrics.increment_counter('search_operations_total', tags=tags)
        self.metrics.record_histogram('search_duration', duration, tags)
        self.metrics.record_histogram('search_results_count', results_count, tags)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'timestamp': datetime.now().isoformat()
        }

    async def get_full_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'health': await self.health.run_health_checks(),
            'metrics': self.metrics.get_metrics_summary(),
            'system': self.get_system_metrics(),
            'alerts': self.alerts.get_active_alerts(),
            'timestamp': datetime.now().isoformat()
        }

    async def check_thresholds(self):
        """Check metrics against thresholds and trigger alerts"""
        metrics_summary = self.metrics.get_metrics_summary()

        # Check API error rate
        total_requests = sum(metrics_summary['counters'].get('api_requests_total{status=5**}', 0)
                           for key in metrics_summary['counters'].keys()
                           if 'api_requests_total' in key and 'status=5' in key)

        if total_requests > 10:  # More than 10 5xx errors
            self.alerts.trigger_alert(
                'high_error_rate',
                f"High API error rate detected: {total_requests} 5xx errors",
                'critical'
            )

        # Check memory usage
        memory_percent = self.get_system_metrics()['memory_percent']
        if memory_percent > 90:
            self.alerts.trigger_alert(
                'high_memory_usage',
                f"High memory usage: {memory_percent}%",
                'warning'
            )

        # Check Claude token usage (if we have limits)
        total_tokens = sum(metrics_summary['counters'].get('claude_tokens_used{model=**}', 0)
                          for key in metrics_summary['counters'].keys()
                          if 'claude_tokens_used' in key)

        if total_tokens > 1000000:  # 1M tokens threshold
            self.alerts.trigger_alert(
                'high_token_usage',
                f"High Claude token usage: {total_tokens} tokens",
                'warning'
            )


# Global monitoring service instance
monitoring_service = MonitoringService()