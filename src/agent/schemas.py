"""
JSON schemas for agent tool calls.

Defines the structured format for agent actions.
The fine-tuned model outputs JSON matching these schemas.
"""

from pydantic import BaseModel, Field


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
