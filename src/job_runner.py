import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.models import Job, JobStatus
from src.cli import utils as cli_utils

DEFAULT_OUTPUT_DIR = "/tmp/evtexture/output"
POLL_INTERVAL = 1.0


def _build_cli_command(input_params: Dict[str, Any], output_dir: str) -> list[str]:
    """Build CLI command from input parameters."""
    input_path = input_params.get("input")
    if not input_path:
        raise ValueError("input parameter is required")

    cmd = [
        sys.executable,
        "-m",
        "src.cli.cli",
        input_path,
        "-o",
        output_dir,
    ]

    if "model" in input_params:
        cmd.extend(["--model", input_params["model"]])
    if input_params.get("download", False):
        cmd.append("--download")
    if "window_size" in input_params:
        cmd.extend(["--window-size", str(input_params["window_size"])])
    if "stride" in input_params:
        cmd.extend(["--stride", str(input_params["stride"])])
    if "fps" in input_params:
        cmd.extend(["--fps", str(input_params["fps"])])
    if "device" in input_params:
        cmd.extend(["--device", input_params["device"]])
    if "video_output" in input_params:
        cmd.extend(["--video-output", input_params["video_output"]])
    if "max_frames" in input_params:
        cmd.extend(["--max-frames", str(input_params["max_frames"])])
    if "start_frame" in input_params:
        cmd.extend(["--start-frame", str(input_params["start_frame"])])
    if input_params.get("deinterlace", False):
        cmd.append("--deinterlace")
    if input_params.get("verbose", False):
        cmd.append("-v")

    return cmd


def _count_output_frames(output_dir: str) -> int:
    """Count the number of frame files in the output directory."""
    if not os.path.exists(output_dir):
        return 0

    frame_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    count = 0
    for f in os.listdir(output_dir):
        if Path(f).suffix.lower() in frame_extensions:
            count += 1
    return count


def _get_total_frames(input_params: Dict[str, Any]) -> Optional[int]:
    """Get the total number of frames in the input."""
    input_path = input_params.get("input")
    if not input_path:
        return None

    try:
        if cli_utils.is_video_file(input_path):
            info = cli_utils.get_video_info(input_path)
            max_frames = input_params.get("max_frames")
            if max_frames is not None:
                return min(info["frame_count"], max_frames)
            return info["frame_count"]
        elif cli_utils.is_frame_directory(input_path):
            frame_files = cli_utils.load_frame_directory(input_path)
            max_frames = input_params.get("max_frames")
            if max_frames is not None:
                return min(len(frame_files), max_frames)
            return len(frame_files)
    except Exception:
        pass

    return None


def _clear_output_dir(output_dir: str) -> None:
    """Clear the output directory before a new run."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


async def run_job(
    job: Job,
    get_status_callback: Callable[[], str],
) -> Dict[str, Any]:
    """Run the EvTexture job as a subprocess.

    Args:
        job: Job object with input parameters
        get_status_callback: Callback to check current job status

    Returns:
        Dict containing result information
    """
    input_params = job.input_params

    output_path = input_params.get("output")
    if output_path:
        output_dir = output_path
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    _clear_output_dir(output_dir)

    cmd = _build_cli_command(input_params, output_dir)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    while True:
        if process.poll() is not None:
            break

        current_status = get_status_callback()
        if current_status == JobStatus.CANCELLED:
            process.kill()
            process.wait()
            return {"output_dir": output_dir, "cancelled": True}

        time.sleep(POLL_INTERVAL)

    returncode = process.returncode

    if returncode == 0:
        final_frame_count = _count_output_frames(output_dir)
        return {
            "output_dir": output_dir,
            "total_frames": final_frame_count,
            "success": True,
        }
    else:
        stdout, stderr = process.communicate()
        return {
            "output_dir": output_dir,
            "success": False,
            "error": stderr or "Process failed with no error output",
        }
