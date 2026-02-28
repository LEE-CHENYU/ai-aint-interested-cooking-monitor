"""
Record a backup demo using pre-recorded video.

Captures the full system running against a video file,
so if the live camera is flaky during judging, you have a fallback.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Record backup demo")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", default="backup_demo.mp4", help="Output recording")
    args = parser.parse_args()

    # TODO: Implement
    # 1. Run the full pipeline against the video file
    # 2. Record the terminal/UI output alongside
    # 3. Save as a single video or screen recording

    print(f"Backup recorder ready. Input: {args.video}, Output: {args.output}")


if __name__ == "__main__":
    main()
