"""
Batch processing API routes for OPZ Product Matcher
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from api.dependencies import get_current_user
from services.batch_service import batch_service, JobPriority
from models.database import User

router = APIRouter()


class JobSubmissionRequest(BaseModel):
    """Request model for job submission"""
    job_type: str = Field(..., description="Type of job to submit")
    payload: Dict[str, Any] = Field(..., description="Job payload data")
    priority: str = Field("normal", description="Job priority (low, normal, high, urgent)")
    max_retries: int = Field(3, description="Maximum number of retries")


class BulkImportRequest(BaseModel):
    """Request model for bulk import"""
    files: List[Dict[str, Any]] = Field(..., description="List of files to import")
    vendor_id: Optional[int] = Field(None, description="Vendor ID for all files")


@router.post("/jobs", response_model=Dict[str, str])
async def submit_job(
    request: JobSubmissionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Submit a new batch job"""
    try:
        # Convert priority string to enum
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }

        priority = priority_map.get(request.priority.lower(), JobPriority.NORMAL)

        job_id = await batch_service.submit_job(
            request.job_type,
            request.payload,
            priority,
            request.max_retries
        )

        return {"job_id": job_id, "message": "Job submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.post("/bulk-import", response_model=Dict[str, str])
async def submit_bulk_import(
    request: BulkImportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Submit a bulk import job"""
    try:
        payload = {
            "files": request.files,
            "vendor_id": request.vendor_id
        }

        job_id = await batch_service.submit_job(
            "bulk_import",
            payload,
            JobPriority.NORMAL,
            3
        )

        return {"job_id": job_id, "message": "Bulk import job submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit bulk import: {str(e)}")


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get job status"""
    try:
        job_data = await batch_service.get_job_status(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        return job_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a job"""
    try:
        success = await batch_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

        return {"message": f"Job {job_id} cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/jobs", response_model=Dict[str, Any])
async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """List jobs with optional filtering"""
    try:
        # Get queued jobs
        queued_jobs = await batch_service.get_job_queue()

        # Get active jobs
        active_jobs = await batch_service.get_active_jobs()

        # Combine and filter
        all_jobs = queued_jobs + active_jobs

        if status:
            status_map = {
                "pending": "pending",
                "running": "running",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "cancelled"
            }
            target_status = status_map.get(status.lower())
            if target_status:
                all_jobs = [job for job in all_jobs if job.get('status') == target_status]

        if job_type:
            all_jobs = [job for job in all_jobs if job.get('job_type') == job_type]

        # Sort by creation time (newest first) and limit
        all_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        all_jobs = all_jobs[:limit]

        return {
            "jobs": all_jobs,
            "total": len(all_jobs),
            "filters": {
                "status": status,
                "job_type": job_type,
                "limit": limit
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/queue", response_model=Dict[str, Any])
async def get_queue_status(current_user: User = Depends(get_current_user)):
    """Get queue status"""
    try:
        queued_jobs = await batch_service.get_job_queue()
        active_jobs = await batch_service.get_active_jobs()

        return {
            "queued_jobs": len(queued_jobs),
            "active_jobs": len(active_jobs),
            "max_concurrent": batch_service.max_concurrent_jobs,
            "queue_details": queued_jobs[:10],  # Show first 10 queued jobs
            "active_details": active_jobs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.post("/document-processing")
async def submit_document_processing(
    document_ids: List[str],
    priority: str = "normal",
    current_user: User = Depends(get_current_user)
):
    """Submit document processing job"""
    try:
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }

        job_priority = priority_map.get(priority.lower(), JobPriority.NORMAL)

        job_id = await batch_service.submit_job(
            "document_processing",
            {"document_ids": document_ids},
            job_priority,
            3
        )

        return {"job_id": job_id, "message": "Document processing job submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit document processing: {str(e)}")


@router.post("/benchmark-update")
async def submit_benchmark_update(
    benchmark_ids: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
):
    """Submit benchmark update job"""
    try:
        job_id = await batch_service.submit_job(
            "benchmark_update",
            {"benchmark_ids": benchmark_ids or []},
            JobPriority.NORMAL,
            3
        )

        return {"job_id": job_id, "message": "Benchmark update job submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit benchmark update: {str(e)}")


@router.post("/product-sync")
async def submit_product_sync(
    vendor_ids: Optional[List[int]] = None,
    current_user: User = Depends(get_current_user)
):
    """Submit product synchronization job"""
    try:
        job_id = await batch_service.submit_job(
            "product_sync",
            {"vendor_ids": vendor_ids or []},
            JobPriority.NORMAL,
            3
        )

        return {"job_id": job_id, "message": "Product sync job submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit product sync: {str(e)}")