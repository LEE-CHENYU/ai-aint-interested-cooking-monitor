"""
JSON schemas for agent tool calls and recipe models.

Defines the structured format for agent actions and recipe steps.
The fine-tuned model outputs JSON matching these schemas.
"""

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Recipe models
# ---------------------------------------------------------------------------


class CompletionType(str, Enum):
    vlm = "vlm"
    timer = "timer"
    user_confirm = "user_confirm"


class RecipeStep(BaseModel):
    id: int
    instruction: str
    completion_type: CompletionType
    vlm_signal: str | None = None
    timer_seconds: int | None = None
    depends_on: list[int] = Field(default_factory=list)
    parallel_group: str | None = None
    status: str = Field(default="pending", pattern="^(pending|active|done)$")


class Recipe(BaseModel):
    dish: str
    servings: int = 2
    estimated_time_minutes: int = 25
    steps: list[RecipeStep]


# ---------------------------------------------------------------------------
# Agent action schemas
# ---------------------------------------------------------------------------


class SetTimerAction(BaseModel):
    tool: str = "set_timer"
    name: str = Field(description="Timer name, e.g. 'pasta'")
    seconds: int = Field(description="Duration in seconds")


class AdjustTimerAction(BaseModel):
    tool: str = "adjust_timer"
    name: str
    delta_seconds: int = Field(description="Positive to extend, negative to shorten")


class SpeakAction(BaseModel):
    tool: str = "speak"
    text: str = Field(description="What to say to the user")
    priority: str = Field(
        description="Alert priority", pattern="^(low|medium|high|critical)$"
    )


class ShowCardAction(BaseModel):
    tool: str = "show_card"
    title: str
    bullets: list[str]


class MarkStepDoneAction(BaseModel):
    tool: str = "mark_step_done"
    step_id: int


class ReorderStepsAction(BaseModel):
    tool: str = "reorder_steps"
    new_order: list[int] = Field(description="Step IDs in new execution order")


class AgentResponse(BaseModel):
    """Complete agent response containing zero or more actions."""

    actions: list[
        SetTimerAction
        | AdjustTimerAction
        | SpeakAction
        | ShowCardAction
        | MarkStepDoneAction
        | ReorderStepsAction
    ] = Field(default_factory=list)
    reasoning: str | None = Field(
        default=None, description="Brief explanation of why these actions were chosen"
    )
