"""
End-to-end demo runner.

Wires together all components for the hackathon demo:
1. Camera ingest (or pre-recorded video)
2. Event detection with temporal smoothing
3. World state management
4. Agent loop with tool execution
5. Phone UI server

Usage:
    python scripts/run_demo.py --recipe configs/recipes/pasta_and_sauce.yaml
    python scripts/run_demo.py --video path/to/recording.mp4  # Backup demo
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the cooking monitor demo")
    parser.add_argument(
        "--recipe",
        default="configs/recipes/pasta_and_sauce.yaml",
        help="Recipe config path",
    )
    parser.add_argument(
        "--zones",
        default="configs/zones.yaml",
        help="Zone config path",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Pre-recorded video path (for backup demo)",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8765,
        help="WebSocket port for phone UI",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  AI Kitchen Cooking Monitor - Demo")
    print("=" * 50)
    print(f"Recipe: {args.recipe}")
    print(f"Zones: {args.zones}")
    print(f"Video: {args.video or 'LIVE CAMERA'}")
    print(f"UI: ws://localhost:{args.ui_port}")
    print("=" * 50)

    # TODO: Wire up all components
    # 1. Load configs
    # 2. Start camera streams (or video playback)
    # 3. Initialize world state with recipe
    # 4. Start agent loop
    # 5. Start UI server
    # 6. Run main loop

    print("Demo runner ready. Implement component wiring.")


if __name__ == "__main__":
    main()
