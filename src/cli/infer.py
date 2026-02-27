import cv2
import os
import torch
import numpy as np
from typing import List, Callable, Optional
from tqdm import tqdm

from src.basicsr.archs.evtexture_arch import EvTexture
from src.cli.esim import EventSimulatorConfig, events_generator
from src.cli.evoxels import package_bidirectional_event_voxels


def load_model(model_path: str, device: str = "cuda") -> EvTexture:
    """Load EvTexture model from checkpoint."""
    model = EvTexture()
    params_dict = torch.load(model_path, map_location=device, weights_only=True)[
        "params_ema"
    ]
    model.load_state_dict(params_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def prepare_frames_for_inference(frames: List[np.ndarray], device: str) -> torch.Tensor:
    """Convert frames to tensor format for inference.

    Args:
        frames: List of frames in BGR format (H, W, C)
        device: Device to put tensor on

    Returns:
        Tensor of shape (1, N, C, H, W) in RGB format, values in [0, 1]
    """
    n = len(frames)
    h, w = frames[0].shape[:2]

    frames_rgb = []
    for frame in frames:
        frame_rgb = frame[:, :, ::-1].copy()
        frame_float = frame_rgb.astype(np.float32) / 255.0
        frames_rgb.append(frame_float)

    frames_tensor = torch.from_numpy(np.stack(frames_rgb, axis=0))
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)

    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    return frames_tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert output tensor to numpy array.

    Args:
        tensor: Tensor of shape (B, N, C, H, W) in RGB format, values in [0, 1]

    Returns:
        List of numpy arrays in BGR format
    """
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = torch.clamp(tensor, min=0, max=1)
    tensor = (tensor * 255).cpu().numpy().astype(np.uint8)

    frames = []
    for i in range(tensor.shape[0]):
        frame_bgr = tensor[i, :, :, ::-1]
        frames.append(frame_bgr)

    return frames


def generate_events(frames: List[np.ndarray], fps: float) -> List[torch.Tensor]:
    """Generate events from frames using eSIM.

    Args:
        frames: List of frames in BGR format
        fps: Frames per second for timestamp calculation

    Returns:
        List of event tensors
    """
    imgs = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        log_img = np.log(gray.astype(np.float32) + 1e-3)
        imgs.append(torch.from_numpy(log_img))

    timestamps = [i / fps for i in range(len(imgs))]
    config = EventSimulatorConfig()

    events_list = list(events_generator(imgs, timestamps, config))
    return events_list


def process_window(
    frames_tensor: torch.Tensor,
    events_list: List[torch.Tensor],
    fps: float,
    device: str,
    model: EvTexture,
    bins: int = 5,
) -> torch.Tensor:
    """Process a single window of frames.

    Args:
        frames_tensor: Tensor of shape (1, N, C, H, W)
        events_list: List of event tensors
        fps: Frames per second
        device: Device to use
        model: EvTexture model
        bins: Number of event bins

    Returns:
        Output tensor of shape (1, N, C, 4H, 4W)
    """
    n, h, w = frames_tensor.shape[1], frames_tensor.shape[3], frames_tensor.shape[4]
    timestamps = [i / fps for i in range(n)]

    all_events = torch.cat([e.to(device) for e in events_list], dim=0)

    if len(all_events) == 0:
        raise ValueError("No events generated from frames")

    xs = all_events[:, 0]
    ys = all_events[:, 1]
    ts = all_events[:, 2]
    pols = all_events[:, 3]

    voxels_f = (
        torch.stack(
            package_bidirectional_event_voxels(
                xs, ys, ts, pols, timestamps, False, bins, (h, w)
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    voxels_b = (
        torch.stack(
            package_bidirectional_event_voxels(
                xs, ys, ts, pols, timestamps, True, bins, (h, w)
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model.forward(frames_tensor, voxels_f, voxels_b)

    return output


def sliding_window_inference(
    frames: List[np.ndarray],
    model: EvTexture,
    window_size: int = 7,
    stride: int = 1,
    fps: float = 24.0,
    device: str = "cuda",
    verbose: bool = True,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[np.ndarray]:
    """Run inference with sliding window.

    Args:
        frames: List of input frames (BGR format)
        model: Loaded EvTexture model
        window_size: Number of frames per window
        stride: Window stride (1 = overlapping, window_size = no overlap)
        fps: Frames per second for event generation
        device: Device to use
        verbose: Show progress
        output_dir: If provided, save frames to this directory incrementally
        progress_callback: Optional callback(completed_frames, total_frames)

    Returns:
        List of upscaled frames (BGR format)
    """
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    scale = 4

    output_h, output_w = h * scale, w * scale

    output_accum = np.zeros((n_frames, output_h, output_w, 3), dtype=np.float32)
    output_counts = np.zeros((n_frames, output_h, output_w, 1), dtype=np.float32)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        saved_frames = [False] * n_frames
    else:
        saved_frames = None

    frames_tensor = prepare_frames_for_inference(frames, device)

    num_windows = max(1, (n_frames - window_size) // stride + 1)

    iterator = range(0, n_frames, stride)
    if verbose:
        iterator = tqdm(iterator, desc="Processing windows")

    for window_start in iterator:
        window_end = min(window_start + window_size, n_frames)
        window_frames = frames_tensor[:, window_start:window_end, :, :, :]

        window_events = generate_events(frames[window_start:window_end], fps)

        if len(window_events) == 0:
            continue

        try:
            output = process_window(
                window_frames,
                window_events,
                fps,
                device,
                model,
            )
        except Exception as e:
            print(f"Warning: Error processing window {window_start}-{window_end}: {e}")
            continue

        output_frames = tensor_to_numpy(output)

        for i, frame_idx in enumerate(range(window_start, window_end)):
            out_frame = output_frames[i]

            output_accum[frame_idx] += out_frame
            output_counts[frame_idx] += 1

        if output_dir is not None:
            for frame_idx in range(window_start, window_end):
                if not saved_frames[frame_idx]:
                    frame = np.clip(output_accum[frame_idx], 0, 255).astype(np.uint8)
                    frame_path = os.path.join(output_dir, f"{frame_idx:08d}.png")
                    cv2.imwrite(frame_path, frame)
                    saved_frames[frame_idx] = True

        if progress_callback is not None:
            completed = sum(1 for s in saved_frames) if saved_frames else 0
            progress_callback(completed, n_frames)

    valid_mask = output_counts > 0
    output_accum = output_accum / np.maximum(output_counts, 1)

    output_frames = []
    for i in range(n_frames):
        if output_counts[i].sum() > 0:
            frame = np.clip(output_accum[i], 0, 255).astype(np.uint8)
        else:
            frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        output_frames.append(frame)

    return output_frames
