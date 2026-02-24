import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm


def get_video_info(video_path: str) -> Dict:
    """Get video information using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def load_video_frames(
    video_path: str,
    deinterlace: bool = False,
    start_frame: int = 0,
    max_frames: int = None,
) -> Tuple[List[np.ndarray], Dict]:
    """Load frames from a video file.

    Args:
        video_path: Path to video file
        deinterlace: If True, apply deinterlacing (not fully implemented - use without for now)
        start_frame: Frame index to start from (default: 0)
        max_frames: Maximum number of frames to load (default: all)

    Returns:
        frames: List of frames as numpy arrays (BGR format)
        info: Video information dict
    """
    import warnings

    info = get_video_info(video_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_idx = start_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_idx += 1
            if max_frames is not None and len(frames) >= max_frames:
                break

        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    return frames, info


def deinterlace_frame(frame: np.ndarray) -> np.ndarray:
    """Placeholder deinterlace function.

    Note: Full deinterlacing requires reading raw video frames before OpenCV
    converts them. For now, this is a placeholder.
    """
    return frame


def load_frame_directory(frame_dir: str) -> List[str]:
    """Load frame file paths from a directory.

    Returns sorted list of image file paths.
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        raise ValueError(f"Not a directory: {frame_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    frame_files = []
    for f in frame_dir.iterdir():
        if f.suffix.lower() in image_extensions:
            frame_files.append(str(f))

    if len(frame_files) == 0:
        raise ValueError(f"No image files found in: {frame_dir}")

    frame_files.sort()
    return frame_files


def load_frames_from_files(frame_files: List[str]) -> List[np.ndarray]:
    """Load frames from a list of file paths."""
    frames = []
    for fp in tqdm(frame_files, desc="Loading frames"):
        frame = cv2.imread(fp)
        if frame is None:
            raise ValueError(f"Failed to read frame: {fp}")
        frames.append(frame)
    return frames


def load_frame_directory_as_arrays(
    frame_dir: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """Load frames from a directory.

    Returns:
        frames: List of frames as numpy arrays (BGR format)
        filenames: List of original filenames
    """
    frame_files = load_frame_directory(frame_dir)
    filenames = [os.path.basename(f) for f in frame_files]
    frames = load_frames_from_files(frame_files)
    return frames, filenames


def save_frames(
    frames: List[np.ndarray], output_dir: str, filenames: Optional[List[str]] = None
) -> None:
    """Save frames to a directory, preserving original filenames.

    Args:
        frames: List of frames (BGR format)
        output_dir: Output directory
        filenames: Optional list of filenames. If None, uses frame_%06d.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filenames is None:
        filenames = [f"frame_{i:06d}.png" for i in range(len(frames))]

    for i, (frame, filename) in enumerate(
        tqdm(zip(frames, filenames), total=len(frames), desc="Saving frames")
    ):
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame)


def encode_video(
    frames: List[np.ndarray], output_path: str, fps: float, codec: str = "mp4v"
) -> None:
    """Encode frames to a video file.

    Args:
        frames: List of frames (BGR format)
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec (default: mp4v)
    """
    if len(frames) == 0:
        raise ValueError("No frames to encode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Failed to create video writer for: {output_path}")

    for frame in tqdm(frames, desc="Encoding video"):
        writer.write(frame)

    writer.release()


def is_video_file(path: str) -> bool:
    """Check if a path is a video file."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    return Path(path).suffix.lower() in video_extensions


def is_frame_directory(path: str) -> bool:
    """Check if a path is a directory containing images."""
    path = Path(path)
    if not path.is_dir():
        return False

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return any(f.suffix.lower() in image_extensions for f in path.iterdir())
