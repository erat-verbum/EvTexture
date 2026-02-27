from enum import Enum
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    status: HealthStatus
    message: str
    timestamp: str
    service_name: str = "evtexture"


class JobStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StartJobRequest(BaseModel):
    job_id: str
    input_params: Dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    input_params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
