"""
Step engine — rule-based step progression.

Manages recipe step state: activates the first eligible step on init,
advances through steps based on VLM detections, timer expiry, or user
confirmation. No LLM needed for step transitions.
"""

import logging

from src.agent.schemas import Recipe, RecipeStep, CompletionType

logger = logging.getLogger(__name__)


class StepEngine:
    """Drives step-by-step recipe progression using deterministic rules."""

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self._steps: dict[int, RecipeStep] = {s.id: s for s in recipe.steps}
        # Activate the first eligible step(s)
        self._advance()

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def get_active_step(self) -> RecipeStep | None:
        """Return the single 'active' step shown to the user (first active)."""
        for step in self.recipe.steps:
            if step.status == "active":
                return step
        return None

    def get_active_steps(self) -> list[RecipeStep]:
        """Return all currently active steps (parallel steps can coexist)."""
        return [s for s in self.recipe.steps if s.status == "active"]

    @property
    def all_done(self) -> bool:
        return all(s.status == "done" for s in self.recipe.steps)

    # ------------------------------------------------------------------
    # Completion triggers
    # ------------------------------------------------------------------

    def check_vlm_result(self, vlm_result: dict) -> list[dict]:
        """
        For VLM-completion steps: check if `step_complete` flag is set.

        Returns a list of action dicts emitted by the state change.
        """
        actions: list[dict] = []
        step_complete = vlm_result.get("step_complete", False)
        if not step_complete:
            return actions

        for step in self.get_active_steps():
            if step.completion_type == CompletionType.vlm:
                actions.extend(self._complete_step(step.id))
        return actions

    def check_timer_expired(self, timer_name: str) -> list[dict]:
        """
        For timer-completion steps: complete the step whose timer expired.

        The timer_name is matched against 'step_{id}' naming convention.
        """
        actions: list[dict] = []
        for step in self.get_active_steps():
            if step.completion_type == CompletionType.timer:
                expected_timer = f"step_{step.id}"
                if timer_name == expected_timer:
                    actions.extend(self._complete_step(step.id))
        return actions

    def user_confirm(self, step_id: int) -> list[dict]:
        """For user_confirm steps: user pressed Enter / tapped Done."""
        step = self._steps.get(step_id)
        if step and step.status == "active" and step.completion_type == CompletionType.user_confirm:
            return self._complete_step(step_id)
        return []

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _complete_step(self, step_id: int) -> list[dict]:
        """Mark a step done and advance to the next eligible step(s)."""
        step = self._steps[step_id]
        step.status = "done"
        actions = [{"type": "step_done", "step_id": step_id, "instruction": step.instruction}]
        logger.info("Step %d done: %s", step_id, step.instruction)

        new_actions = self._advance()
        actions.extend(new_actions)

        if self.all_done:
            actions.append({"type": "recipe_done", "dish": self.recipe.dish})

        return actions

    def _advance(self) -> list[dict]:
        """Find next pending step(s) with all deps satisfied and activate them."""
        actions: list[dict] = []
        done_ids = {s.id for s in self.recipe.steps if s.status == "done"}

        for step in self.recipe.steps:
            if step.status != "pending":
                continue
            # Check all dependencies are done
            if all(dep in done_ids for dep in step.depends_on):
                step.status = "active"
                logger.info("Step %d activated: %s", step.id, step.instruction)
                actions.append({
                    "type": "step_activated",
                    "step_id": step.id,
                    "instruction": step.instruction,
                    "completion_type": step.completion_type.value,
                })
                # If it's a timer step, emit a start_timer action
                if step.completion_type == CompletionType.timer and step.timer_seconds:
                    actions.append({
                        "type": "start_timer",
                        "timer_name": f"step_{step.id}",
                        "seconds": step.timer_seconds,
                        "step_id": step.id,
                    })

        return actions
