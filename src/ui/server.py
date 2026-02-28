"""
WebSocket server for phone UI.

Serves a local web page that the phone connects to on the same network.
Pushes real-time updates: current step, timers, agent cues.
Voice output via Web Speech API on the phone browser.
"""

import asyncio
import json

import websockets


class UIServer:
    """WebSocket server for phone display."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Handle messages from phone (e.g., user taps "step done")
                data = json.loads(message)
                print(f"Phone message: {data}")
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, data: dict):
        """Send state update to all connected phones."""
        message = json.dumps(data)
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients]
            )

    async def send_cue(self, text: str, priority: str = "medium"):
        """Send a voice/visual cue to the phone."""
        await self.broadcast({
            "type": "cue",
            "text": text,
            "priority": priority,
            "speak": priority in ("high", "critical"),
        })

    async def send_state(self, state: dict):
        """Send updated world state to the phone."""
        await self.broadcast({"type": "state_update", **state})

    async def start(self):
        print(f"UI server starting on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # Run forever
