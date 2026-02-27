# EvTexture: Event-driven Texture Enhancement for Video Super-Resolution

Purpose: Video super-resolution CLI tool with FastAPI web service interface for job management. Supports both CLI and HTTP API usage.

## Development Workflow

1. **Clarify requirements**: Ask clarifying questions of the user to understand the task fully.
2. **Plan the approach**: Create/update a TODO list to outline the steps needed to complete the task.
3. **Research**: Use context7 to look up how libraries work when needed for the task.
4. **Implement**: Make targeted, small changes, one-by-one to ensure quality and avoid errors.
5. **Verify**: Read the modified files to ensure the changes are correct.
6. **Lint and Type Check**: Run linting (`make lint`), fix linting issues (`make lint-fix`), and type checking (`make check`) to ensure code quality.
7. **Test**: Run tests to verify functionality. Then run tests (`make test` for all, `make test-unit`/`make test-int` for specific types).
8. **Complete**: Do not stop until all tasks on the TODO list are completed and verified.

## Rules of Engagement

1. Think incredibly hard and long before getting to the Implement step, writing lots and lots, considering all possible options and then choosing the right one
2. Be concise specifically when responding to the user that a task has been completed

## Makefile Usage

Before running any command, read the relevant Makefile.

### Service-level commands

- `make install`: uv venv/sync
- `make lint lint-fix check`: ruff/pyright
- `make test test-unit test-int`: pytest
- `make run`: uvicorn

## File Tree

```
src/
├── __init__.py           # Package init
├── __main__.py           # CLI entry point (python -m evtexture)
├── main.py               # FastAPI entry point with HTTP endpoints
├── models.py             # Pydantic models for requests, responses, and data structures
├── job_runner.py         # Job execution logic - runs CLI as subprocess, tracks progress
├── cli/                  # CLI implementation
│   ├── __init__.py
│   ├── cli.py            # Main CLI command
│   ├── infer.py          # Inference logic
│   ├── utils.py          # Utility functions
│   ├── download.py       # Model downloading
│   ├── esim.py           # Event simulator
│   └── evoxels.py        # Event voxel packaging
└── basicsr/              # BasicSR library (arch, ops, etc.)

test/
├── unit/                 # Unit tests
│   └── test__<name_of_file_being_tested>__<name_of_feature_being_tested>.py
└── integration/         # Integration tests
    └── test__<name_of_file_being_tested>__<name_of_feature_being_tested>.py
```

## Project Folder Structure

Each service follows this folder structure:

```
evtexture/
├── Dockerfile            # Container configuration with GPU support
├── docker-compose.yml    # Local development environment
├── pyproject.toml        # Project configuration
├── Makefile              # Common commands
├── .env.sample           # Environment variables template
├── src/                  # All source code
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models and data structures
│   ├── job_runner.py     # Job execution - runs CLI as subprocess
│   ├── cli/              # CLI implementation
│   └── basicsr/          # BasicSR library
└── test/
    ├── unit/             # Unit tests
    └── integration/      # Integration tests
```

## Service Components

- **Dockerfile**: Container configuration with GPU support (`--gpus all`)
- **uv**: Package manager (installed in local `.venv`)
- **ruff**: Linting and formatting
- **pyright**: Type checking
- **FastAPI**: HTTP interface for job management
- **pyproject.toml**: Project configuration
- **Python `subprocess`**: Running CLI as background process for job tracking
- **Pydantic**: For data validation and serialization - Pydantic models should be defined for every non-simple object
- **Type annotations**: All method parameters and return types must be annotated for better code quality and IDE support
- **Docstrings**: Each method must include a docstring with: a description, Args section (parameter names, types, descriptions), Returns section (return type and description), and Raises section (exceptions and when they're raised). Format example:
  ```
  def method_name(self, param1: Type) -> ReturnType:
      """
      Brief description of the method.
      
      Args:
          param1 (Type): Description of parameter
      
      Returns:
          ReturnType: Description of return value
      
      Raises:
          ExceptionType: Description of when this exception is raised
      """
  ```
- **Class method organization**: Public methods in a class should always be written at the bottom of the class AFTER all of the private methods (those starting with underscore). This improves code readability by grouping implementation details together.
- **Test naming**: Test files should be named:
  - `test__<name_of_file_being_tested>.py` for simple cases where the file contains tests for a single feature
  - `test__<name_of_file_being_tested>__<name_of_feature_being_tested>.py` when the file would become too large or contain tests for multiple distinct features

## CLI Usage

```bash
evtexture <input> -o <output> [options]
```

Options:
- `-o, --output`: Output directory or video file (required)
- `--model`: Model choice (REDS, Vimeo90K)
- `--download`: Auto-download model if missing
- `--window-size`: Frames per inference window (default: 7)
- `--stride`: Window stride (default: 1)
- `--fps`: Input video FPS for event generation
- `--device`: Device to use (cuda, cpu)
- `--video-output`: Encode output as video file
- `--max-frames`: Maximum number of frames to process
- `--start-frame`: Starting frame index
- `--deinterlace`: Deinterlace input video
- `-v, --verbose`: Show verbose output

## HTTP Interface

### Health Check

- `GET /health`
  - **Response**:
    ```json
    {
      "status": "healthy|unhealthy|degraded",
      "message": "string",
      "timestamp": "ISO timestamp",
      "service_name": "string"
    }
    ```

### Job Management

- `POST /job`
  - **Request Body**:
    ```json
    {
      "job_id": "string",
      "input_params": {
        "input": "/path/to/input",
        "output": "/path/to/output",
        "model": "Vimeo90K",
        "window_size": 7,
        "stride": 1,
        "fps": 24.0,
        "device": "cuda",
        "download": true
      }
    }
    ```
  - **Response**: Job object with status, progress, timestamps
  - **Errors**: 409 if job already running

- `GET /job`
  - **Response**: Job object or `null` if no job exists
    ```json
    {
      "id": "string",
      "status": "running|completed|failed|cancelled",
      "progress": 0-100,
      "result": { ... },
      "error": "string",
      "created_at": "ISO timestamp",
      "started_at": "ISO timestamp",
      "finished_at": "ISO timestamp"
    }
    ```

- `POST /job/cancel`
  - **Request Body**: `{}` (empty)
  - **Response**: `{"message": "Job cancelled"}`
  - **Errors**: 404 if no job, 400 if job not running

## Output Directory Handling

- If user specifies `output` in input_params, use that path
- If not specified, use `/tmp/evtexture/output/` (cleared before each run)
- Frames are saved incrementally during processing (not all at end) to manage memory

## Dockerfile Requirements

The Dockerfile must include:
```dockerfile
RUN make install && \
    make lint && \
    make check
```

And include GPU support:
```dockerfile
--gpus all
```

## Key Implementation Details

1. **Incremental Frame Saving**: The inference code saves frames to disk after each processing window to prevent memory exhaustion with high-resolution video.

2. **Job Progress Tracking**: Progress is calculated by polling the output directory for completed frames:
   - `progress = (completed_frames / total_frames) * 100`

3. **Subprocess Management**: The job runner spawns the CLI as a subprocess and tracks its state:
   - Running: subprocess is alive
   - Completed: subprocess finished with exit code 0
   - Failed: subprocess finished with non-zero exit code
   - Cancelled: subprocess was killed
