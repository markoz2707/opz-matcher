"""
Monitoring API routes for OPZ Product Matcher
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from api.dependencies import get_current_user
from services.monitoring_service import monitoring_service
from models.database import User

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    is_healthy = await monitoring_service.health.is_healthy()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": monitoring_service.metrics.get_metrics_summary()["timestamp"]
    }


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status(current_user: User = Depends(get_current_user)):
    """Get complete system status (requires authentication)"""
    try:
        return await monitoring_service.get_full_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(current_user: User = Depends(get_current_user)):
    """Get performance metrics (requires authentication)"""
    try:
        return monitoring_service.metrics.get_metrics_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/alerts", response_model=Dict[str, Any])
async def get_alerts(
    severity: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get active alerts (requires authentication)"""
    try:
        alerts = monitoring_service.alerts.get_active_alerts(severity_filter=severity)
        return {
            "alerts": alerts,
            "total": len(alerts),
            "severity_filter": severity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.delete("/alerts/{alert_id}")
async def clear_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """Clear a specific alert (requires authentication)"""
    try:
        monitoring_service.alerts.clear_alert(alert_id)
        return {"message": f"Alert {alert_id} cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear alert: {str(e)}")


@router.post("/check-thresholds")
async def check_thresholds(current_user: User = Depends(get_current_user)):
    """Manually trigger threshold checks (requires authentication)"""
    try:
        await monitoring_service.check_thresholds()
        return {"message": "Threshold checks completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check thresholds: {str(e)}")