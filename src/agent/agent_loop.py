"""
Main agent loop.

Runs on a 2-5 second cycle (or on state change):
1. Build compact prompt from world state
2. Query local model for actions
3. Validate actions against schema
4. Execute via ToolExecutor

This is where "agentic" behavior lives — the model decides and acts autonomously.
"""

import json
import time

from src.agent.schemas import AgentResponse
from src.agent.tools import ToolExecutor


class AgentLoop:
    """Core agent decision loop."""

    def __init__(self, world_state, model=None, interval_seconds: float = 3.0):
        self.world_state = world_state
        self.model = model  # Local Gemma model
        self.interval = interval_seconds
        self.executor = ToolExecutor(world_state)
        self.running = False

    def build_prompt(self) -> str:
        """Build a compact prompt from current world state."""
        state_snapshot = self.world_state.get_snapshot()
        return json.dumps(state_snapshot, indent=None)

    def query_model(self, prompt: str) -> AgentResponse:
        """Query the local model and parse response."""
        if self.model is None:
            # Placeholder: return empty response when no model loaded
            return AgentResponse(actions=[])

        # TODO: Implement model inference
        # 1. Format prompt for FunctionGemma / Gemma
        # 2. Run inference
        # 3. Parse JSON output
        # 4. Validate with AgentResponse schema
        return AgentResponse(actions=[])

    def step(self):
        """Run one agent decision cycle."""
        prompt = self.build_prompt()
        response = self.query_model(prompt)
        results = self.executor.execute(response)
        return results

    def run(self):
        """Run the agent loop continuously."""
        self.running = True
        print(f"Agent loop started (interval: {self.interval}s)")

        while self.running:
            try:
                results = self.step()
                if results:
                    print(f"Agent actions: {results}")
            except Exception as e:
                print(f"Agent loop error: {e}")

            time.sleep(self.interval)

    def stop(self):
        self.running = False
