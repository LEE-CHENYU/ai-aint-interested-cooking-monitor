"""
Main agent loop — single-threaded, VLM-driven, rule-based step progression.

Each cycle:
1. CAPTURE: get next frame from sequencer
2. INFER: VLM detects dish state + step completion
3. UPDATE STATE: translate VLM signals into world state
4. SAFETY: check deterministic rules + VLM safe flag
5. SMOOTH: temporal smoothing on step_complete → step engine
6. TIMER CHECK: expired timers → step engine
7. USER CONFIRM: blocking input() for user_confirm steps
8. PROCESS ACTIONS: announce step transitions via TTS
"""

import logging
import time

import numpy as np

from src.agent.safety_engine import SafetyEngine
from src.agent.schemas import CompletionType
from src.agent.step_engine import StepEngine
from src.agent.tts import TTSEngine
from src.agent.tools import ToolExecutor
from src.perception.detector import VLMDetector
from src.perception.image_sequencer import ImageSequencer
from src.perception.temporal_smoother import TemporalSmoother
from src.world_state.state import WorldState
from src.world_state.timer_engine import TimerEngine

# Dummy frame for mock mode when sequencer has no images
_DUMMY_FRAME = np.zeros((224, 224, 3), dtype=np.uint8)

logger = logging.getLogger(__name__)


class AgentLoop:
    """Single-threaded agentic cooking loop."""

    def __init__(
        self,
        step_engine: StepEngine,
        detector: VLMDetector,
        sequencer: ImageSequencer,
        smoother: TemporalSmoother,
        tts: TTSEngine,
        timer_engine: TimerEngine,
        safety_engine: SafetyEngine | None,
        world_state: WorldState,
        auto_confirm: bool = False,
    ):
        self.step_engine = step_engine
        self.detector = detector
        self.sequencer = sequencer
        self.smoother = smoother
        self.tts = tts
        self.timer_engine = timer_engine
        self.safety_engine = safety_engine
        self.world_state = world_state
        self.tool_executor = ToolExecutor(tts)
        self.auto_confirm = auto_confirm

    def run(self):
        """Run the full recipe loop."""
        # Announce first step
        active = self.step_engine.get_active_step()
        if active:
            self.tts.speak(f"Let's start! Step {active.id}: {active.instruction}", "medium")

        # Process initial actions (timers for auto-activated steps)
        for step in self.step_engine.get_active_steps():
            if step.completion_type == CompletionType.timer and step.timer_seconds:
                timer_name = f"step_{step.id}"
                self.timer_engine.create_timer(timer_name, step.timer_seconds)

        max_cycles = 500  # safeguard against infinite loops
        cycle = 0
        while not self.step_engine.all_done and cycle < max_cycles:
            self._cycle()
            cycle += 1
            # Small delay so timers can tick
            time.sleep(0.5)

        if self.step_engine.all_done:
            self.tts.speak("Recipe complete! Enjoy your meal!", "high")
        else:
            self.tts.speak("Demo ended (cycle limit reached).", "medium")

    def _cycle(self):
        """Run one perception-action cycle."""
        active_step = self.step_engine.get_active_step()

        # ------------------------------------------------------------------
        # 1. CAPTURE
        # ------------------------------------------------------------------
        frame, filename = self.sequencer.next()
        if frame is None:
            if self.detector.mock:
                # Mock mode doesn't need real frames — use a dummy
                frame = _DUMMY_FRAME
            else:
                # Real mode: no more images — handle remaining non-VLM steps
                self._handle_non_vlm_steps()
                return

        # ------------------------------------------------------------------
        # 2. INFER
        # ------------------------------------------------------------------
        step_signal = active_step.vlm_signal if active_step else None
        vlm_result = self.detector.detect(frame, step_signal)
        logger.info("VLM: %s", vlm_result)

        # ------------------------------------------------------------------
        # 3. UPDATE STATE — bridge VLM output to world state flat signals
        # ------------------------------------------------------------------
        zone_signals = self._vlm_to_zone_signals(vlm_result)
        self.world_state.update_zone("stove", zone_signals)

        # ------------------------------------------------------------------
        # 4. SAFETY — deterministic rules + VLM safe flag
        # ------------------------------------------------------------------
        self._check_safety(vlm_result)

        # ------------------------------------------------------------------
        # 5. SMOOTH — temporal smoothing on step_complete
        # ------------------------------------------------------------------
        actions: list[dict] = []
        step_complete = vlm_result.get("step_complete", False)
        smoothed = self.smoother.update("step_complete", step_complete)
        if smoothed.stable and smoothed.value is True:
            new_actions = self.step_engine.check_vlm_result(vlm_result)
            actions.extend(new_actions)
            self.smoother.reset("step_complete")

        # ------------------------------------------------------------------
        # 6. TIMER CHECK
        # ------------------------------------------------------------------
        expired = self.timer_engine.get_expired()
        for timer in expired:
            timer_actions = self.step_engine.check_timer_expired(timer.name)
            actions.extend(timer_actions)
            self.timer_engine.cancel_timer(timer.name)

        # ------------------------------------------------------------------
        # 7. USER CONFIRM — blocking input() or auto-confirm in mock mode
        # ------------------------------------------------------------------
        active_step = self.step_engine.get_active_step()
        if active_step and active_step.completion_type == CompletionType.user_confirm:
            if self.auto_confirm:
                self.tts.speak(
                    f"Step {active_step.id}: {active_step.instruction} — auto-confirmed.",
                    "medium",
                )
            else:
                self.tts.speak(
                    f"Step {active_step.id}: {active_step.instruction} — Press Enter when done.",
                    "medium",
                )
                input()  # Blocking
            confirm_actions = self.step_engine.user_confirm(active_step.id)
            actions.extend(confirm_actions)

        # ------------------------------------------------------------------
        # 8. PROCESS ACTIONS
        # ------------------------------------------------------------------
        self._process_actions(actions)

    def _handle_non_vlm_steps(self):
        """Handle remaining timer/user_confirm steps when images are exhausted."""
        # Check timers
        expired = self.timer_engine.get_expired()
        for timer in expired:
            actions = self.step_engine.check_timer_expired(timer.name)
            self._process_actions(actions)
            self.timer_engine.cancel_timer(timer.name)

        # Check user confirm
        active_step = self.step_engine.get_active_step()
        if active_step and active_step.completion_type == CompletionType.user_confirm:
            if self.auto_confirm:
                self.tts.speak(
                    f"Step {active_step.id}: {active_step.instruction} — auto-confirmed.",
                    "medium",
                )
            else:
                self.tts.speak(
                    f"Step {active_step.id}: {active_step.instruction} — Press Enter when done.",
                    "medium",
                )
                input()
            actions = self.step_engine.user_confirm(active_step.id)
            self._process_actions(actions)

    def _vlm_to_zone_signals(self, vlm_result: dict) -> dict:
        """Translate VLM output to flat zone signals for SafetyEngine."""
        state = vlm_result.get("state", "unknown")
        safe = vlm_result.get("safe", True)
        reason = vlm_result.get("reason", "").lower()
        return {
            # Map boil_over back to "boiling" so existing rules still match,
            # and add a dedicated boil_over flag for explicit detection
            "water_state": "boiling" if state == "boil_over" else state,
            "boil_over_detected": state == "boil_over",
            "pot_present": state != "unknown",
            "smoke_suspected": not safe and "smoke" in reason,
            "steam_level": "high" if state in ("boiling", "boil_over") else "low",
        }

    def _check_safety(self, vlm_result: dict):
        """Run safety checks from both the rule engine and VLM safe flag."""
        # VLM-reported unsafe condition
        if not vlm_result.get("safe", True):
            reason = vlm_result.get("reason", "unsafe condition detected")
            self.tts.speak(f"Safety alert: {reason}", "critical")

        # Deterministic safety rules
        if self.safety_engine:
            flat_signals = self.world_state.get_flat_signals()
            violations = self.safety_engine.check_all(flat_signals, self.world_state)
            if violations:
                forced = self.safety_engine.escalate_if_needed(violations, [])
                self.tool_executor.execute_forced(forced)
                for v in violations:
                    self.world_state.record_alert(v.rule_id)

    def _process_actions(self, actions: list[dict]):
        """Process action dicts emitted by StepEngine."""
        for action in actions:
            match action["type"]:
                case "step_done":
                    self.tts.speak(
                        f"Step {action['step_id']} complete: {action['instruction']}",
                        "medium",
                    )
                case "step_activated":
                    self.tts.speak(
                        f"Next — Step {action['step_id']}: {action['instruction']}",
                        "medium",
                    )
                case "start_timer":
                    self.timer_engine.create_timer(
                        action["timer_name"], action["seconds"]
                    )
                    self.tts.speak(
                        f"Timer started: {action['seconds']}s for step {action['step_id']}",
                        "low",
                    )
                case "recipe_done":
                    self.tts.speak(
                        f"{action['dish']} is ready!",
                        "high",
                    )
