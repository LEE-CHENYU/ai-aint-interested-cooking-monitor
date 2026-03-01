"""
Demo runner — single-threaded agentic cooking loop.

Usage:
    # Mock mode (no camera, no GPU needed)
    python scripts/run_demo.py --recipe configs/recipes/pasta_and_sauce.yaml --mock

    # With real VLM model + demo images
    python scripts/run_demo.py --recipe configs/recipes/pasta_and_sauce.yaml \
        --images data/demo_sequences/ --model google/gemma-3-4b-it

    # No audio (console only)
    python scripts/run_demo.py --mock --no-audio
"""

import argparse
import logging
import sys

from src.agent.agent_loop import AgentLoop
from src.agent.recipe_loader import load_recipe
from src.agent.safety_engine import SafetyEngine
from src.agent.step_engine import StepEngine
from src.agent.tts import TTSEngine
from src.perception.detector import VLMDetector
from src.perception.image_sequencer import ImageSequencer
from src.perception.temporal_smoother import TemporalSmoother
from src.world_state.state import WorldState
from src.world_state.timer_engine import TimerEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def main():
    parser = argparse.ArgumentParser(description="Run the cooking monitor demo")
    parser.add_argument(
        "--recipe",
        default="configs/recipes/pasta_and_sauce.yaml",
        help="Recipe YAML path",
    )
    parser.add_argument(
        "--images",
        default="data/demo_sequences/",
        help="Directory of demo images (sorted by filename)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model path (e.g. google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock VLM detector (no GPU needed)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable TTS audio (console output only)",
    )
    parser.add_argument(
        "--safety-rules",
        default="configs/safety_rules.yaml",
        help="Safety rules YAML path",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast demo: 2s timers + auto-confirm user steps (implies --mock)",
    )
    args = parser.parse_args()

    if args.fast:
        args.mock = True

    print("=" * 55)
    print("  AI Kitchen Cooking Monitor - Demo")
    print("=" * 55)
    print(f"  Recipe:      {args.recipe}")
    print(f"  Images:      {args.images}")
    print(f"  VLM:         {'MOCK' if args.mock else args.model or 'NONE'}")
    print(f"  Audio:       {'OFF' if args.no_audio else 'ON'}")
    print(f"  Safety:      {args.safety_rules}")
    print(f"  Fast mode:   {'ON' if args.fast else 'OFF'}")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------------
    recipe = load_recipe(args.recipe)

    # Fast mode: override all timer steps to 2 seconds
    if args.fast:
        for step in recipe.steps:
            if step.timer_seconds is not None:
                step.timer_seconds = 2

    timer_engine = TimerEngine()
    world_state = WorldState(recipe=recipe)
    step_engine = StepEngine(recipe)
    sequencer = ImageSequencer(args.images)
    detector = VLMDetector(model_path=args.model, mock=args.mock)
    smoother = TemporalSmoother()
    tts = TTSEngine(audio=not args.no_audio)
    safety_engine = SafetyEngine(rules_path=args.safety_rules)

    agent = AgentLoop(
        step_engine=step_engine,
        detector=detector,
        sequencer=sequencer,
        smoother=smoother,
        tts=tts,
        timer_engine=timer_engine,
        safety_engine=safety_engine,
        world_state=world_state,
        auto_confirm=args.fast,
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
