import click
import time
import tempfile
import os
import subprocess
from pathlib import Path

from cli import download as download_module, infer, utils


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output directory or video file")
@click.option(
    "--model",
    default="Vimeo90K",
    type=click.Choice(["REDS", "Vimeo90K"]),
    help="Model choice",
)
@click.option(
    "--download",
    "download",
    is_flag=True,
    default=False,
    help="Auto-download model if missing",
)
@click.option("--window-size", default=7, type=int, help="Frames per inference window")
@click.option(
    "--stride",
    default=1,
    type=int,
    help="Window stride (1=overlap, window_size=no overlap)",
)
@click.option(
    "--fps", default=24.0, type=float, help="Input video FPS for event generation"
)
@click.option(
    "--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use"
)
@click.option("--video-output", default=None, help="Encode output as video file")
@click.option(
    "--max-frames",
    default=None,
    type=int,
    help="Maximum number of frames to process (default: all frames)",
)
@click.option(
    "--start-frame",
    default=0,
    type=int,
    help="Starting frame index (default: 0)",
)
@click.option(
    "--deinterlace",
    is_flag=True,
    default=False,
    help="Deinterlace input video using ffmpeg (for interlaced sources)",
)
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
def main(
    input: str,
    output: str,
    model: str,
    download: bool,
    window_size: int,
    stride: int,
    fps: float,
    device: str,
    video_output: str,
    max_frames: int,
    start_frame: int,
    deinterlace: bool,
    verbose: bool,
):
    """EvTexture: Event-driven Texture Enhancement for Video Super-Resolution.

    INPUT: Video file or directory containing image frames.
    """
    start_time = time.time()

    if verbose:
        click.echo("=" * 60)
        click.echo("EvTexture CLI - Video Super-Resolution")
        click.echo("=" * 60)

    if verbose:
        click.echo(f"\n[1/5] Loading input: {input}")

    input_path = Path(input)
    temp_deinterlaced = None

    if utils.is_video_file(input):
        if deinterlace:
            if verbose:
                click.echo("  - Deinterlacing with ffmpeg (yadif)...")
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_deinterlaced = temp_file.name
            temp_file.close()
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input,
                "-vf",
                "yadif=mode=0",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                temp_deinterlaced,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            input = temp_deinterlaced
            if verbose:
                click.echo("  - Deinterlacing complete")

        frames, video_info = utils.load_video_frames(
            input, start_frame=start_frame, max_frames=max_frames
        )
        if video_output is None:
            video_output = str(Path(output).with_suffix(".mp4"))
        if fps is None or fps == 24.0:
            fps = video_info["fps"]
        filenames = None
        if verbose:
            click.echo(f"  - Loaded {len(frames)} frames from video")
            click.echo(f"  - Resolution: {video_info['width']}x{video_info['height']}")
            click.echo(f"  - FPS: {video_info['fps']}")
            click.echo(f"  - Starting from frame: {start_frame}")
            if max_frames is not None:
                click.echo(f"  - Limited to {max_frames} frames")
            if deinterlace:
                click.echo(f"  - Deinterlaced: Yes (ffmpeg)")
    elif utils.is_frame_directory(input):
        frames, filenames = utils.load_frame_directory_as_arrays(input)
        if max_frames is not None and len(frames) > max_frames:
            frames = frames[:max_frames]
            filenames = filenames[:max_frames] if filenames else None
            if verbose:
                click.echo(f"  - Limited to {max_frames} frames")
        if verbose:
            click.echo(f"  - Loaded {len(frames)} frames from directory")
            click.echo(f"  - Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    else:
        raise click.ClickException(
            f"Input must be a video file or directory containing images"
        )

    if verbose:
        click.echo(f"\n[2/5] Loading model: {model}")

    model_path = download_module.ensure_model(
        model, auto_download=download, verbose=verbose
    )
    if verbose:
        click.echo(f"  - Model path: {model_path}")

    if verbose:
        click.echo(f"\n[3/5] Loading EvTexture model to {device}")

    evtexture_model = infer.load_model(str(model_path), device=device)

    if verbose:
        click.echo(f"\n[4/5] Running inference...")
        click.echo(f"  - Window size: {window_size}")
        click.echo(f"  - Stride: {stride}")
        click.echo(f"  - FPS: {fps}")

    output_frames = infer.sliding_window_inference(
        frames,
        evtexture_model,
        window_size=window_size,
        stride=stride,
        fps=fps,
        device=device,
        verbose=verbose,
    )

    if verbose:
        click.echo(f"\n[5/5] Saving output")

    output_path = Path(output)
    if output_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        utils.encode_video(output_frames, str(output_path), fps=fps)
        if verbose:
            click.echo(f"  - Saved video: {output_path}")
    else:
        output_dir = output_path
        utils.save_frames(output_frames, str(output_dir), filenames)
        if verbose:
            click.echo(f"  - Saved {len(output_frames)} frames to: {output_dir}")

    if video_output and video_output != str(output_path):
        utils.encode_video(output_frames, video_output, fps=fps)
        if verbose:
            click.echo(f"  - Saved video: {video_output}")

    elapsed = time.time() - start_time
    click.echo(f"\n✓ Complete in {elapsed:.1f}s")
    click.echo(
        f"  Input: {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}"
    )
    click.echo(
        f"  Output: {len(output_frames)} frames at {output_frames[0].shape[1]}x{output_frames[0].shape[0]}"
    )

    if temp_deinterlaced and os.path.exists(temp_deinterlaced):
        os.unlink(temp_deinterlaced)


if __name__ == "__main__":
    main()
