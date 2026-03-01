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
import threading
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

# Color palette for mock state visualization (BGR for OpenCV)
_STATE_COLORS = {
    "cold":      (180, 130, 70),    # steel blue
    "simmering": (70, 160, 230),    # warm orange
    "boiling":   (60, 80, 220),     # red
    "boil_over": (30, 30, 200),     # deep red
    "done":      (100, 180, 80),    # green
    "unknown":   (100, 100, 100),   # grey
}

logger = logging.getLogger(__name__)


def _make_mock_frame(vlm_result: dict) -> np.ndarray:
    """Generate a synthetic 480x640 frame visualizing the mock VLM state."""
    import cv2

    state = vlm_result.get("state", "unknown")
    safe = vlm_result.get("safe", True)
    reason = vlm_result.get("reason", "")

    # Background color from state
    bg = _STATE_COLORS.get(state, _STATE_COLORS["unknown"])
    frame = np.full((480, 640, 3), bg, dtype=np.uint8)

    # Add noise for texture
    noise = np.random.randint(-15, 15, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw state label
    label = state.upper().replace("_", " ")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
    x = (640 - tw) // 2
    y = 220
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255, 255, 255), 4, cv2.LINE_AA)

    # Draw reason below
    (rw, rh), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    rx = (640 - rw) // 2
    cv2.putText(frame, reason, (rx, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (220, 220, 220), 2, cv2.LINE_AA)

    # Safety warning flash
    if not safe:
        cv2.rectangle(frame, (0, 380), (640, 480), (0, 0, 255), -1)
        warn = "UNSAFE: " + reason
        (ww, wh), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        wx = (640 - ww) // 2
        cv2.putText(frame, warn, (wx, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Simulated pot circle
    cv2.circle(frame, (320, 340), 60, (200, 200, 200), 3, cv2.LINE_AA)

    # Bubbles for boiling states
    if state in ("boiling", "boil_over"):
        for _ in range(12):
            bx = 320 + np.random.randint(-50, 50)
            by = 340 + np.random.randint(-50, 30)
            br = np.random.randint(3, 10)
            cv2.circle(frame, (bx, by), br, (255, 255, 255), -1, cv2.LINE_AA)

    return frame


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
        ui_server=None,
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
        self.ui_server = ui_server
        self._confirm_event = threading.Event()
        self._confirmed_step_id = None
        self._frame_index = 0
        self._total_frames = len(self.sequencer)
        if ui_server:
            ui_server.on_user_confirm = self._on_ui_confirm

    def _on_ui_confirm(self, step_id: int):
        """Callback from UI thread when user taps Done."""
        self._confirmed_step_id = step_id
        self._confirm_event.set()

    def _wait_for_confirm(self):
        """Block until user confirms via console Enter or UI Done button."""
        self._confirm_event.clear()

        # Spawn daemon thread for console fallback
        def _console_input():
            try:
                input()
            except EOFError:
                pass
            self._confirm_event.set()

        t = threading.Thread(target=_console_input, daemon=True)
        t.start()
        self._confirm_event.wait()

    def run(self):
        """Run the full recipe loop."""
        # Announce first step
        active = self.step_engine.get_active_step()
        if active:
            self.tts.speak(f"Let's start! Step {active.id}: {active.instruction}", "medium")
            total = len(self.step_engine.recipe.steps)
            if self.ui_server:
                self.ui_server.fire_step(
                    active.id, total, active.instruction,
                    active.completion_type.value, self.step_engine.recipe.dish,
                )

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
            if self.ui_server:
                self.ui_server.fire_done(self.step_engine.recipe.dish)
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
        t0 = time.time()
        vlm_result = self.detector.detect(frame, step_signal)
        latency_ms = int((time.time() - t0) * 1000)
        logger.info("VLM: %s", vlm_result)

        # Send frame + VLM result to demo UI
        self._frame_index += 1
        if self.ui_server:
            # In mock mode, generate a synthetic visualization frame
            display_frame = _make_mock_frame(vlm_result) if self.detector.mock else frame
            self.ui_server.fire_image(
                display_frame, filename or f"mock_{self._frame_index:03d}",
                self._frame_index, self._total_frames or self._frame_index,
            )
            self.ui_server.fire_vlm_result(vlm_result, latency_ms)

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
        # 7. USER CONFIRM — blocking wait or auto-confirm in mock mode
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
                self._wait_for_confirm()
            confirm_actions = self.step_engine.user_confirm(active_step.id)
            actions.extend(confirm_actions)

        # ------------------------------------------------------------------
        # 8. PROCESS ACTIONS
        # ------------------------------------------------------------------
        self._process_actions(actions)

        # ------------------------------------------------------------------
        # 9. SEND TIMER UPDATES to UI
        # ------------------------------------------------------------------
        if self.ui_server:
            for timer in self.timer_engine.get_active():
                self.ui_server.fire_timer(
                    timer.name,
                    int(timer.remaining_seconds),
                    timer.duration_seconds,
                )

    def _handle_non_vlm_steps(self):
        """Handle remaining timer/user_confirm steps when images are exhausted."""
        # Check timers
        expired = self.timer_engine.get_expired()
        for timer in expired:
            actions = self.step_engine.check_timer_expired(timer.name)
            self._process_actions(actions)
            self.timer_engine.cancel_timer(timer.name)

        # Send timer updates to UI
        if self.ui_server:
            for timer in self.timer_engine.get_active():
                self.ui_server.fire_timer(
                    timer.name,
                    int(timer.remaining_seconds),
                    timer.duration_seconds,
                )

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
                self._wait_for_confirm()
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
            if self.ui_server:
                self.ui_server.fire_safety(reason, "critical")

        # Deterministic safety rules
        if self.safety_engine:
            flat_signals = self.world_state.get_flat_signals()
            violations = self.safety_engine.check_all(flat_signals, self.world_state)
            if violations:
                forced = self.safety_engine.escalate_if_needed(violations, [])
                self.tool_executor.execute_forced(forced)
                for v in violations:
                    self.world_state.record_alert(v.rule_id)
                    if self.ui_server:
                        self.ui_server.fire_safety(v.message, v.severity)

    def _process_actions(self, actions: list[dict]):
        """Process action dicts emitted by StepEngine."""
        total = len(self.step_engine.recipe.steps)
        dish = self.step_engine.recipe.dish
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
                    if self.ui_server:
                        self.ui_server.fire_step(
                            action["step_id"], total, action["instruction"],
                            action["completion_type"], dish,
                        )
                case "start_timer":
                    self.timer_engine.create_timer(
                        action["timer_name"], action["seconds"]
                    )
                    self.tts.speak(
                        f"Timer started: {action['seconds']}s for step {action['step_id']}",
                        "low",
                    )
                    if self.ui_server:
                        self.ui_server.fire_timer(
                            action["timer_name"], action["seconds"],
                            action["seconds"],
                        )
                case "recipe_done":
                    self.tts.speak(
                        f"{action['dish']} is ready!",
                        "high",
                    )
                    if self.ui_server:
                        self.ui_server.fire_done(action["dish"])
