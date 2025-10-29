#!/usr/bin/env python3
#
# create_video.py - Create video from rendered frames with intelligent ordering
#

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Add current directory to Python path for utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def frames2video_sorted(render_path, output_path, cam_num=16, fps=25, mode="timeline"):
    """
    Generate video with intelligent frame ordering based on camera count and desired mode.

    Args:
        render_path: Path to directory containing rendered frames
        output_path: Output video file path
        cam_num: Number of camera poses (viewpoints)
        fps: Video frame rate
        mode: Ordering mode - "timeline" (fixed camera, time changes)
                            or "camera_rotation" (fixed time, camera changes)
    """
    # Get all PNG files and sort by name
    images = [img for img in os.listdir(render_path) if img.endswith(".png")]
    images.sort()  # Sort by filename: 00000.png, 00001.png, ...

    # Calculate total frames and verify data consistency
    total_frames = len(images) // cam_num
    if len(images) % cam_num != 0:
        print(f"Warning: Total images ({len(images)}) not divisible by camera count ({cam_num})")
        print(f"Using {total_frames} frames (truncated)")

    print(f"Total images: {len(images)}, Cameras: {cam_num}, Frames: {total_frames}")
    print(f"Ordering mode: {mode}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(render_path, images[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {images[0]}")

    height, width, layers = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_written = 0

    if mode == "timeline":
        # Timeline mode: Fixed camera, time changes
        # Shows temporal evolution from a single viewpoint
        for cam_idx in range(cam_num):
            for frame_idx in range(total_frames):
                img_index = frame_idx * cam_num + cam_idx
                if img_index < len(images):
                    frame_path = os.path.join(render_path, images[img_index])
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        video.write(frame)
                        frames_written += 1
                        # print(f"Written: Camera {cam_idx}, Frame {frame_idx} -> {images[img_index]}")
                    else:
                        print(f"Warning: Could not read {frame_path}")
                else:
                    print(f"Warning: Index {img_index} out of range")

    elif mode == "camera_rotation":
        # Camera rotation mode: Fixed time, camera changes
        # Shows different viewpoints at the same time moment
        for frame_idx in range(total_frames):
            for cam_idx in range(cam_num):
                img_index = frame_idx * cam_num + cam_idx
                if img_index < len(images):
                    frame_path = os.path.join(render_path, images[img_index])
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        video.write(frame)
                        frames_written += 1
                        print(f"Written: Frame {frame_idx}, Camera {cam_idx} -> {images[img_index]}")
                    else:
                        print(f"Warning: Could not read {frame_path}")
                else:
                    print(f"Warning: Index {img_index} out of range")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    video.release()

    if frames_written == 0:
        raise ValueError("No frames were written to video")

    print(f"Video created successfully: {output_path}")
    print(f"Total frames written: {frames_written}")
    print(f"Video duration: {frames_written / fps:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Create video from rendered frames with intelligent ordering based on camera count"
    )
    parser.add_argument(
        "--render_path", type=str, required=True,
        help="Directory path containing rendered frames (PNG format)"
    )
    parser.add_argument(
        "--output_video", type=str, default="output_video.mp4",
        help="Output video file path (default: output_video.mp4)"
    )
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Video frame rate in frames per second (default: 25)"
    )
    parser.add_argument(
        "--cam_num", type=int, required=True,
        help="Number of camera poses/viewpoints (e.g., 16 for 16 different camera angles)"
    )
    parser.add_argument(
        "--mode", choices=["timeline", "camera_rotation"], default="timeline",
        help="Video ordering mode: 'timeline' (fixed camera, time changes) or 'camera_rotation' (fixed time, camera changes)"
    )

    args = parser.parse_args()

    # Validate input path exists
    if not os.path.exists(args.render_path):
        print(f"Error: Render path does not exist: {args.render_path}")
        sys.exit(1)

    # Check for PNG files in the directory
    png_files = list(Path(args.render_path).glob("*.png"))
    if not png_files:
        print(f"Error: No PNG files found in {args.render_path}")
        sys.exit(1)

    # Print configuration summary
    print("Video Creation Configuration:")
    print(f"  Input path: {args.render_path}")
    print(f"  Output video: {args.output_video}")
    print(f"  Frame rate: {args.fps} FPS")
    print(f"  Camera poses: {args.cam_num}")
    print(f"  Ordering mode: {args.mode}")
    print(f"  Found {len(png_files)} PNG files")

    try:
        # Create video with specified ordering
        frames2video_sorted(
            args.render_path,
            args.output_video,
            args.cam_num,
            args.fps,
            args.mode
        )
        print(f"Video creation completed: {args.output_video}")

    except Exception as e:
        print(f"Error creating video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()