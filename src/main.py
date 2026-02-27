import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from src.job_runner import run_job
from src.models import (
    HealthCheckResponse,
    HealthStatus,
    Job,
    JobStatus,
    StartJobRequest,
)

app = FastAPI(
    title="EvTexture Service",
    description="Event-driven Texture Enhancement for Video Super-Resolution",
    version="0.1.0",
)

_current_job: Optional[Dict[str, Any]] = None
_job_task: Optional[asyncio.Task] = None


def reset_job() -> None:
    """Reset job state. Used for testing."""
    global _current_job, _job_task
    _current_job = None
    _job_task = None


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint that returns service status."""
    return HealthCheckResponse(
        status=HealthStatus.HEALTHY,
        message="Service is running normally",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/job", response_model=Job)
async def start_job(request: StartJobRequest) -> Job:
    """Start a new EvTexture processing job."""
    global _current_job, _job_task

    if _current_job and _current_job.get("status") == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="A job is already running")

    now = datetime.now().isoformat()
    _current_job = {
        "id": request.job_id,
        "status": JobStatus.RUNNING,
        "progress": 0,
        "input_params": request.input_params,
        "result": None,
        "error": None,
        "created_at": now,
        "started_at": now,
        "finished_at": None,
    }

    async def run_and_update() -> None:
        job_obj = Job(
            id=_current_job["id"],
            status=_current_job["status"],
            progress=_current_job["progress"],
            input_params=_current_job["input_params"],
        )
        try:
            result = await run_job(
                job_obj,
                lambda: _current_job["status"] if _current_job else "cancelled",
            )
            if _current_job:
                if result.get("cancelled", False):
                    _current_job["status"] = JobStatus.CANCELLED
                elif result.get("success", False):
                    _current_job["status"] = JobStatus.COMPLETED
                    _current_job["progress"] = 100
                    _current_job["result"] = {
                        "output_dir": result.get("output_dir"),
                        "total_frames": result.get("total_frames"),
                    }
                else:
                    _current_job["status"] = JobStatus.FAILED
                    _current_job["error"] = result.get("error", "Unknown error")
                _current_job["finished_at"] = datetime.now().isoformat()
        except Exception as e:
            if _current_job:
                _current_job["status"] = JobStatus.FAILED
                _current_job["error"] = str(e)
                _current_job["finished_at"] = datetime.now().isoformat()

    _job_task = asyncio.create_task(run_and_update())
    return Job(**_current_job)


@app.get("/job", response_model=Optional[Job])
async def get_job() -> Optional[Job]:
    """Get the current job status."""
    if _current_job is None:
        return None
    return Job(**_current_job)


@app.post("/job/cancel")
async def cancel_job() -> Dict[str, str]:
    """Cancel the current running job."""
    global _current_job, _job_task

    if _current_job is None:
        raise HTTPException(status_code=404, detail="No job found")

    if _current_job["status"] != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is not running")

    _current_job["status"] = JobStatus.CANCELLED
    _current_job["finished_at"] = datetime.now().isoformat()

    if _job_task and not _job_task.done():
        _job_task.cancel()

    return {"message": "Job cancelled"}
