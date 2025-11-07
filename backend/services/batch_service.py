"""
Batch processing service for OPZ Product Matcher
Handles background tasks and job queues using Redis/Celery
"""
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import json
from loguru import logger

from config.settings import settings
from services.cache_service import cache_service
from services.monitoring_service import monitoring_service


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class BatchJob:
    """Represents a batch job"""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.payload = payload
        self.priority = priority
        self.max_retries = max_retries
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.retry_count = 0
        self.error_message: Optional[str] = None
        self.progress = 0.0
        self.result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type,
            'payload': self.payload,
            'priority': self.priority.value,
            'max_retries': self.max_retries,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'error_message': self.error_message,
            'progress': self.progress,
            'result': self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create job from dictionary"""
        job = cls(
            job_id=data['job_id'],
            job_type=data['job_type'],
            payload=data['payload'],
            priority=JobPriority(data['priority']),
            max_retries=data['max_retries']
        )
        job.status = JobStatus(data['status'])
        job.created_at = datetime.fromisoformat(data['created_at'])
        job.started_at = datetime.fromisoformat(data['started_at']) if data['started_at'] else None
        job.completed_at = datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
        job.retry_count = data['retry_count']
        job.error_message = data['error_message']
        job.progress = data['progress']
        job.result = data['result']
        return job


class BatchService:
    """Main batch processing service"""

    def __init__(self):
        self.job_handlers: Dict[str, Callable[[BatchJob], Awaitable[Any]]] = {}
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_queue: List[BatchJob] = []
        self.max_concurrent_jobs = 5
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize batch service"""
        if self._running:
            return

        # Register default job handlers
        self.register_handler('document_processing', self._handle_document_processing)
        self.register_handler('bulk_import', self._handle_bulk_import)
        self.register_handler('benchmark_update', self._handle_benchmark_update)
        self.register_handler('product_sync', self._handle_product_sync)

        self._running = True
        # Start background processing in a separate thread with new event loop
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        def run_background_task():
            """Run the background task in a new event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._process_queue())
            finally:
                loop.close()

        # Run in thread pool to avoid blocking
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(run_background_task)
        executor.shutdown(wait=False)

        logger.info("Batch service initialized")

    async def shutdown(self):
        """Shutdown batch service"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Batch service shutdown")

    def register_handler(self, job_type: str, handler: Callable[[BatchJob], Awaitable[Any]]):
        """Register a job handler"""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered job handler for type: {job_type}")

    async def submit_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3
    ) -> str:
        """Submit a new job"""
        job_id = str(uuid.uuid4())
        job = BatchJob(job_id, job_type, payload, priority, max_retries)

        # Store job in cache
        await cache_service.set(f"job:{job_id}", job.to_dict(), ttl_seconds=86400)  # 24 hours

        # Add to queue
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda j: (-j.priority.value, j.created_at))  # Higher priority first

        monitoring_service.metrics.increment_counter('jobs_submitted_total', tags={'job_type': job_type})
        logger.info(f"Job submitted: {job_id} ({job_type})")

        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        # Check active jobs first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()

        # Check cache
        cached = await cache_service.get(f"job:{job_id}")
        if cached:
            return cached

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            await cache_service.set(f"job:{job_id}", job.to_dict())
            monitoring_service.metrics.increment_counter('jobs_cancelled_total')
            return True

        # Check queue
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self.job_queue.pop(i)
                await cache_service.set(f"job:{job_id}", job.to_dict())
                monitoring_service.metrics.increment_counter('jobs_cancelled_total')
                return True

        return False

    async def get_job_queue(self) -> List[Dict[str, Any]]:
        """Get current job queue"""
        return [job.to_dict() for job in self.job_queue]

    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get active jobs"""
        return [job.to_dict() for job in self.active_jobs.values()]

    async def _process_queue(self):
        """Process job queue"""
        while self._running:
            try:
                # Check if we can start new jobs
                if len(self.active_jobs) < self.max_concurrent_jobs and self.job_queue:
                    job = self.job_queue.pop(0)
                    self.active_jobs[job.job_id] = job
                    # Execute job in background thread with new event loop
                    from concurrent.futures import ThreadPoolExecutor

                    def run_job_task():
                        """Run the job task in a new event loop"""
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self._execute_job(job))
                        finally:
                            loop.close()

                    # Run in thread pool to avoid blocking
                    executor = ThreadPoolExecutor(max_workers=1)
                    executor.submit(run_job_task)
                    executor.shutdown(wait=False)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _execute_job(self, job: BatchJob):
        """Execute a single job"""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            await cache_service.set(f"job:{job.job_id}", job.to_dict())

            monitoring_service.metrics.increment_counter('jobs_started_total', tags={'job_type': job.job_type})

            # Get handler
            handler = self.job_handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job.job_type}")

            # Execute job with monitoring
            with monitoring_service.metrics.time_execution(f"job_{job.job_type}"):
                result = await handler(job)

            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            job.progress = 100.0

            monitoring_service.metrics.increment_counter('jobs_completed_total', tags={'job_type': job.job_type})

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.retry_count += 1
            job.error_message = str(e)

            if job.retry_count < job.max_retries:
                # Retry
                job.status = JobStatus.PENDING
                self.job_queue.append(job)
                self.job_queue.sort(key=lambda j: (-j.priority.value, j.created_at))
                monitoring_service.metrics.increment_counter('jobs_retried_total', tags={'job_type': job.job_type})
            else:
                # Failed permanently
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                monitoring_service.metrics.increment_counter('jobs_failed_total', tags={'job_type': job.job_type})

                # Trigger alert
                monitoring_service.alerts.trigger_alert(
                    'job_failed',
                    f"Job {job.job_id} ({job.job_type}) failed after {job.retry_count} retries: {e}",
                    'error',
                    {'job_id': job.job_id, 'job_type': job.job_type, 'error': str(e)}
                )

        finally:
            # Update cache and remove from active jobs
            await cache_service.set(f"job:{job.job_id}", job.to_dict())
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    # Default job handlers

    async def _handle_document_processing(self, job: BatchJob) -> Dict[str, Any]:
        """Handle document processing job"""
        from services.document_processor import document_processor

        document_ids = job.payload.get('document_ids', [])
        results = []

        for i, doc_id in enumerate(document_ids):
            try:
                # Update progress
                job.progress = (i / len(document_ids)) * 100
                await cache_service.set(f"job:{job.job_id}", job.to_dict())

                # Process document
                result = await document_processor.process_document(doc_id)
                results.append({'document_id': doc_id, 'success': True, 'result': result})

            except Exception as e:
                results.append({'document_id': doc_id, 'success': False, 'error': str(e)})

        return {'processed': len(results), 'results': results}

    async def _handle_bulk_import(self, job: BatchJob) -> Dict[str, Any]:
        """Handle bulk import job"""
        from services.document_processor import document_processor

        files = job.payload.get('files', [])
        results = []

        for i, file_info in enumerate(files):
            try:
                # Update progress
                job.progress = (i / len(files)) * 100
                await cache_service.set(f"job:{job.job_id}", job.to_dict())

                # Import file
                result = await document_processor.process_uploaded_file(
                    file_info['file_path'],
                    file_info['vendor_id'],
                    file_info['document_type']
                )
                results.append({'file': file_info['file_path'], 'success': True, 'result': result})

            except Exception as e:
                results.append({'file': file_info['file_path'], 'success': False, 'error': str(e)})

        return {'imported': len(results), 'results': results}

    async def _handle_benchmark_update(self, job: BatchJob) -> Dict[str, Any]:
        """Handle benchmark update job"""
        # Placeholder - implement benchmark update logic
        await asyncio.sleep(2)  # Simulate work
        return {'message': 'Benchmark update completed', 'updated_records': 42}

    async def _handle_product_sync(self, job: BatchJob) -> Dict[str, Any]:
        """Handle product synchronization job"""
        # Placeholder - implement product sync logic
        await asyncio.sleep(1)  # Simulate work
        return {'message': 'Product sync completed', 'synced_products': 156}


# Global batch service instance
batch_service = BatchService()