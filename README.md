# [EvTexture (ICML 2024)](https://proceedings.mlr.press/v235/kai24a.html)

Official Pytorch implementation for the "EvTexture: Event-driven Texture Enhancement for Video Super-Resolution" paper (ICML 2024).

<p align="center">
    🌐 <a href="https://dachunkai.github.io/evtexture.github.io/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2406.13457" target="_blank">Paper</a> | 🖼️ <a href="https://docs.google.com/presentation/d/1nbDb39TFb374DzBwdz5v20kIREUA0nBH/edit?usp=sharing" target="_blank">Poster</a> <br>
</p>

**Authors**: [Dachun Kai](https://github.com/DachunKai/)<sup>[:email:️](mailto:dachunkai@mail.ustc.edu.cn)</sup>, Jiayao Lu, [Yueyi Zhang](https://scholar.google.com.hk/citations?user=LatWlFAAAAAJ&hl=zh-CN&oi=ao)<sup>[:email:️](mailto:zhyuey@ustc.edu.cn)</sup>, [Xiaoyan Sun](https://scholar.google.com/citations?user=VRG3dw4AAAAJ&hl=zh-CN), *University of Science and Technology of China*

**Feel free to ask questions. If our work helps, please don't hesitate to give us a :star:!**

> **[News]** An extension of this work, **EvTexture++**, is currently under review. The source code and pre-trained models for EvTexture++ will be released in this repository soon.

## :rocket: News
<!-- - [ ] Provide a script for inference on the user's own video -->
- [x] 2024/07/02: Release the colab file for a quick test
- [x] 2024/06/28: Release details to prepare datasets
- [x] 2024/06/08: Publish docker image
- [x] 2024/06/08: Release pretrained models and test sets for quick testing
- [x] 2024/06/07: Video demos released
- [x] 2024/05/25: Initialize the repository
- [x] 2024/05/02: :tada: :tada: Our paper was accepted in ICML'2024

## :bookmark: Table of Content
1. [Video Demos](#video-demos)
2. [Differences from Upstream](#differences-from-upstream)
3. [Code](#code)
   - [Quick Start](#quick-start)
   - [Installation](#installation)
   - [Development Commands](#development-commands)
   - [FastAPI Service](#fastapi-service)
   - [CLI Usage](#cli-usage)
4. [Citation](#citation)
5. [Contact](#contact)
6. [License and Acknowledgement](#license-and-acknowledgement)

## :fire: Video Demos
A $4\times$ upsampling results on the [Vid4](https://paperswithcode.com/sota/video-super-resolution-on-vid4-4x-upscaling) and [REDS4](https://paperswithcode.com/dataset/reds) test sets.

https://github.com/DachunKai/EvTexture/assets/66354783/fcf48952-ea48-491c-a4fb-002bb2d04ad3

https://github.com/DachunKai/EvTexture/assets/66354783/ea3dd475-ba8f-411f-883d-385a5fdf7ff6

https://github.com/DachunKai/EvTexture/assets/66354783/e1e6b340-64b3-4d94-90ee-54f025f255fb

https://github.com/DachunKai/EvTexture/assets/66354783/01880c40-147b-4c02-8789-ced0c1bff9c4

## Differences from Upstream

This repository is a wrapper around the original [EvTexture](https://github.com/DachunKai/EvTexture) research project, adding a FastAPI web service and CLI interface for easier deployment. Key changes from the upstream:

### Architecture
- **FastAPI Service**: Added web service for job management via HTTP API (port 8001)
- **CLI Tool**: Added `evtexture` command-line interface for processing videos
- **Package Manager**: Replaced Conda/pip with [uv](https://github.com/astral-sh/uv) for faster dependency management
- **Python Version**: Requires Python 3.10+ (upstream used 3.7)

### Project Structure
- **Directory Layout**: Moved source code under `src/` directory for proper packaging
- **Build System**: Added `pyproject.toml` with uv build backend (removed `setup.py`)
- **Development Tools**: Added Makefile for common operations and Dockerfile with GPU support

### Usage Changes
- **Installation**: Use `make install` or `uv sync` instead of Conda/pip setup
- **Running**: Use `make run` to start the FastAPI service
- **Job Management**: Process videos via HTTP API endpoints instead of script-based execution
- **Progress Tracking**: Real-time progress via API polling

The core inference code in `src/basicsr/` remains largely unchanged from the upstream repository.

## Code
### Quick Start

```bash
# Install dependencies
make install

# Run the FastAPI service
make run
```

The service runs on port 8001 by default.

### Installation
* Dependencies: Python 3.10+, [uv](https://github.com/astral-sh/uv)

* Using Make (recommended):

    ```bash
    make install
    ```

* Using uv directly:

    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv sync
    ```

* Run in Docker:

  ```bash
  docker-compose up -d
  ```

  Or build and run manually:
  ```bash
  make docker-build
  make docker-run
  ```

### Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Create virtual environment and sync dependencies |
| `make lint` | Run ruff linting |
| `make lint-fix` | Auto-fix linting issues |
| `make check` | Run pyright type checking |
| `make run` | Start FastAPI service on port 8001 |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container with GPU support |
| `make up` | Start services with docker-compose |
| `make down` | Stop docker-compose services |

### FastAPI Service

The project includes a FastAPI web service for job management. The service runs on port 8001.

#### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy|unhealthy|degraded",
  "message": "string",
  "timestamp": "ISO timestamp",
  "service_name": "evtexture"
}
```

#### Job Management

**Create Job:**
```bash
POST /job
```

Request:
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

**Get Job Status:**
```bash
GET /job
```

Response:
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

**Cancel Job:**
```bash
POST /job/cancel
```

Request: `{}` (empty body)

Response: `{"message": "Job cancelled"}`

### Test
1. Download the pretrained models from ([Releases](https://github.com/DachunKai/EvTexture/releases) / [Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab) / [Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)) and place them to `experiments/pretrained_models/EvTexture/`. The network architecture code is in [evtexture_arch.py](https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py).
    * *EvTexture_REDS_BIx4.pth*: trained on REDS dataset with BI degradation for $4\times$ SR scale.
    * *EvTexture_Vimeo90K_BIx4.pth*: trained on Vimeo-90K dataset with BI degradation for $4\times$ SR scale.

2. Download the preprocessed test sets (including events) for REDS4 and Vid4 from ([Releases](https://github.com/DachunKai/EvTexture/releases) / [Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab) / [Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)), and place them to `datasets/`.
    * *Vid4_h5*: HDF5 files containing preprocessed test datasets for Vid4.

    * *REDS4_h5*: HDF5 files containing preprocessed test datasets for REDS4.

3. Run the following command:
    * Test on Vid4 for 4x VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvTexture/test_EvTexture_Vid4_BIx4.yml
      ```
    * Test on REDS4 for 4x VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvTexture/test_EvTexture_REDS4_BIx4.yml
      ```
      This will generate the inference results in `results/`. The output results on REDS4 and Vid4 can be downloaded from ([Releases](https://github.com/DachunKai/EvTexture/releases) / [Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab) / [Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)).

### Data Preparation
* Both video and event data are required as input, as shown in the [snippet](https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py#L70). We package each video and its event data into an [HDF5](https://docs.h5py.org/en/stable/quick.html#quick) file.

* Example: The structure of `calendar.h5` file from the Vid4 dataset is shown below.

  ```arduino
  calendar.h5
  ├── images
  │   ├── 000000 # frame, ndarray, [H, W, C]
  │   ├── ...
  ├── voxels_f
  │   ├── 000000 # forward event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ├── voxels_b
  │   ├── 000000 # backward event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ```
* To simulate and generate the event voxels, refer to the dataset preparation details in [DataPreparation.md](https://github.com/DachunKai/EvTexture/blob/main/datasets/DataPreparation.md).

### Inference on your own video

We provide a CLI tool for processing your own videos. The CLI automatically handles:
- Loading video frames from file or directory
- Generating event data using eSIM event simulation
- Auto-downloading pretrained models
- Processing with sliding window for long videos
- Optional deinterlacing for interlaced sources

The CLI is available after running `make install` (or `uv sync`).

#### Basic Usage

```bash
# Process a video file
evtexture input.mp4 -o output_frames/

# Process with verbose output
evtexture input.mp4 -o output_frames/ --verbose

# Auto-download model if missing
evtexture input.mp4 -o output_frames/ --download

# Use specific model (REDS or Vimeo90K)
evtexture input.mp4 -o output_frames/ --model REDS
```

#### Advanced Options

```bash
# Process interlaced video (e.g., DVD content)
evtexture input.mkv -o output/ --deinterlace

# Limit number of frames
evtexture input.mp4 -o output/ --max-frames 100

# Process specific frame range
evtexture input.mp4 -o output/ --start-frame 50 --max-frames 100

# Adjust window size for memory/speed tradeoff
evtexture input.mp4 -o output/ --window-size 7 --stride 1

# Output as video file
evtexture input.mp4 -o output.mp4
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | required | Output directory or video file |
| `--model` | Vimeo90K | Model choice (REDS or Vimeo90K) |
| `--download` | False | Auto-download model if missing |
| `--window-size` | 7 | Frames per inference window |
| `--stride` | 1 | Window stride (1=overlap, window_size=no overlap) |
| `--fps` | 24.0 | Input video FPS for event generation |
| `--device` | cuda | Device (cuda or cpu) |
| `--video-output` | None | Encode output as video file |
| `--max-frames` | None | Maximum frames to process |
| `--start-frame` | 0 | Starting frame index |
| `--deinterlace` | False | Deinterlace input video |
| `-v, --verbose` | False | Show verbose output |

#### Models

Models are automatically downloaded to `experiments/pretrained_models/EvTexture/`:
- `EvTexture_REDS_BIx4.pth` - Better for lower resolution / detail-heavy content
- `EvTexture_Vimeo90K_BIx4.pth` - Fewer artifacts for larger images

## :blush: Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :smiley:
```
@inproceedings{kai2024evtexture,
  title={{E}v{T}exture: {E}vent-driven {T}exture {E}nhancement for {V}ideo {S}uper-{R}esolution},
  author={Kai, Dachun and Lu, Jiayao and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={22817--22839},
  year={2024},
  volume={235},
  publisher={PMLR}
}
```

## Contact
If you meet any problems, please describe them in issues or contact:
* Dachun Kai: <dachunkai@mail.ustc.edu.cn>

## License and Acknowledgement
This project is released under the Apache-2.0 license. Our work is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR), which is an open source toolbox for image/video restoration tasks. Thanks to the inspirations and codes from [RAFT](https://github.com/princeton-vl/RAFT), [event_utils](https://github.com/TimoStoff/event_utils) and [EvTexture-jupyter](https://github.com/camenduru/EvTexture-jupyter).
