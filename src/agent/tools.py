"""
Tool execution layer.

Reduced role: handles safety engine forced-action execution.
Most action execution is now in agent_loop._process_actions().
Speech is routed through TTSEngine.
"""

import logging

from src.agent.tts import TTSEngine

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes forced safety actions."""

    def __init__(self, tts: TTSEngine, ui_server=None):
        self.tts = tts
        self.ui_server = ui_server

    def execute_forced(self, forced_actions: list[dict]):
        """Execute forced safety actions (raw dicts from SafetyEngine)."""
        results = []
        for fa in forced_actions:
            text = fa.get("text", "")
            priority = fa.get("priority", "critical")
            self.tts.speak(text, priority)
            results.append({"tool": "speak", "status": "ok", "source": "safety_engine"})
        return results
