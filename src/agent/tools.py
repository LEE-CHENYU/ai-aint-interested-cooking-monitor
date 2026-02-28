"""
Tool execution layer.

Implements the actual side effects for each agent tool call:
- set_timer / adjust_timer -> TimerEngine
- speak -> TTS or phone UI
- show_card -> Phone UI via WebSocket
- mark_step_done / reorder_steps -> World state
"""

from src.agent.schemas import AgentResponse


class ToolExecutor:
    """Executes validated tool calls from the agent."""

    def __init__(self, world_state, ui_server=None):
        self.world_state = world_state
        self.ui_server = ui_server

    def execute(self, response: AgentResponse):
        """Execute all actions in an agent response."""
        results = []
        for action in response.actions:
            result = self._execute_one(action)
            results.append(result)
        return results

    def _execute_one(self, action) -> dict:
        """Execute a single tool call."""
        match action.tool:
            case "set_timer":
                self.world_state.timer_engine.create_timer(
                    action.name, action.seconds
                )
                return {"tool": "set_timer", "status": "ok", "name": action.name}

            case "adjust_timer":
                self.world_state.timer_engine.adjust_timer(
                    action.name, action.delta_seconds
                )
                return {"tool": "adjust_timer", "status": "ok", "name": action.name}

            case "speak":
                # TODO: Route to TTS engine or phone UI
                print(f"[SPEAK:{action.priority}] {action.text}")
                return {"tool": "speak", "status": "ok"}

            case "show_card":
                # TODO: Push to phone UI via WebSocket
                print(f"[CARD] {action.title}: {action.bullets}")
                return {"tool": "show_card", "status": "ok"}

            case "mark_step_done":
                self.world_state.mark_step_done(action.step_id)
                return {"tool": "mark_step_done", "status": "ok", "step_id": action.step_id}

            case "reorder_steps":
                self.world_state.reorder_steps(action.new_order)
                return {"tool": "reorder_steps", "status": "ok"}

            case _:
                return {"tool": action.tool, "status": "error", "reason": "unknown tool"}
