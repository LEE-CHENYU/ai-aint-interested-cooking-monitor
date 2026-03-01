"""
WebSocket + HTTP server for phone UI.

Serves the phone UI (index.html) over HTTP and pushes real-time
updates over WebSocket. Handles user actions from the phone
(e.g., tapping "Done" on a step).

HTTP: http://host:port/         -> serves index.html
WS:   ws://host:port/ws        -> real-time updates

Message protocol (server -> phone):
  {"type": "step", "step_id": N, "total_steps": N, "instruction": "...", "completion_type": "vlm|timer|user_confirm", "dish": "..."}
  {"type": "timer", "name": "...", "remaining_seconds": N, "total_seconds": N}
  {"type": "safety", "message": "...", "severity": "critical|high"}
  {"type": "done", "dish": "..."}

Message protocol (phone -> server):
  {"type": "user_confirm", "step_id": N}
"""

import asyncio
import json
import os
from http import HTTPStatus
from pathlib import Path

import websockets


UI_DIR = Path(__file__).parent


class UIServer:
    """WebSocket + HTTP server for phone display."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.on_user_confirm = None  # Callback: fn(step_id: int)

    async def handler(self, websocket):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                print(f"Phone: {data}")
                if data.get("type") == "user_confirm" and self.on_user_confirm:
                    self.on_user_confirm(data.get("step_id"))
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, data: dict):
        """Send a message to all connected phones."""
        message = json.dumps(data)
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients]
            )

    async def send_step(self, step_id: int, total_steps: int,
                        instruction: str, completion_type: str,
                        dish: str = ""):
        """Notify phone of a new active step."""
        await self.broadcast({
            "type": "step",
            "step_id": step_id,
            "total_steps": total_steps,
            "instruction": instruction,
            "completion_type": completion_type,
            "dish": dish,
        })

    async def send_timer(self, name: str, remaining_seconds: int,
                         total_seconds: int):
        """Send timer update to the phone."""
        await self.broadcast({
            "type": "timer",
            "name": name,
            "remaining_seconds": remaining_seconds,
            "total_seconds": total_seconds,
        })

    async def send_safety(self, message: str,
                          severity: str = "critical"):
        """Send a safety alert to the phone."""
        await self.broadcast({
            "type": "safety",
            "message": message,
            "severity": severity,
        })

    async def send_done(self, dish: str):
        """Notify phone that the recipe is complete."""
        await self.broadcast({
            "type": "done",
            "dish": dish,
        })

    async def process_request(self, path, request_headers):
        """Serve index.html for HTTP GET requests."""
        if path == "/" or path == "/index.html":
            html_path = UI_DIR / "index.html"
            if html_path.exists():
                body = html_path.read_bytes()
                return (
                    HTTPStatus.OK,
                    [("Content-Type", "text/html; charset=utf-8"),
                     ("Content-Length", str(len(body)))],
                    body,
                )
        if path == "/ws":
            return None  # Let websockets handle WebSocket upgrade
        return None  # Default: let websockets handle it

    async def start(self):
        """Start the combined HTTP + WebSocket server."""
        print(f"UI server starting:")
        print(f"  Phone UI:  http://{self.host}:{self.port}/")
        print(f"  WebSocket: ws://{self.host}:{self.port}/ws")

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            process_request=self.process_request,
        ):
            await asyncio.Future()  # Run forever


async def _demo():
    """Demo mode: run server and send sample messages."""
    server = UIServer()

    async def send_demo_sequence():
        await asyncio.sleep(2)
        print("\nSending demo sequence...")

        # Step 1: VLM step
        await server.send_step(1, 6, "Dice the tofu into 1-inch cubes",
                               "vlm", "mapo_tofu")
        await asyncio.sleep(4)

        # Step 2: User confirm
        await server.send_step(2, 6, "Mince garlic, ginger, and scallions",
                               "user_confirm", "mapo_tofu")
        await asyncio.sleep(4)

        # Step 3: VLM step
        await server.send_step(3, 6, "Heat oil in wok until shimmering",
                               "vlm", "mapo_tofu")
        await asyncio.sleep(4)

        # Step 4: Timer step
        await server.send_step(4, 6, "Add doubanjiang and cook for 30 seconds",
                               "timer", "mapo_tofu")
        await asyncio.sleep(1)
        await server.send_timer("step_4_timer", 30, 30)
        await asyncio.sleep(5)

        # Safety alert
        await server.send_safety("Boil-over detected! Reduce heat immediately!")
        await asyncio.sleep(5)

        # Step 5: Timer step
        await server.send_step(5, 6, "Add tofu and broth, simmer for 5 minutes",
                               "timer", "mapo_tofu")
        await asyncio.sleep(1)
        await server.send_timer("step_5_timer", 15, 300)
        await asyncio.sleep(5)

        # Step 6: VLM step
        await server.send_step(6, 6,
                               "Add cornstarch slurry and stir until thickened",
                               "vlm", "mapo_tofu")
        await asyncio.sleep(5)

        # Done!
        await server.send_done("mapo_tofu")
        print("Demo sequence complete.")

    await asyncio.gather(
        server.start(),
        send_demo_sequence(),
    )


if __name__ == "__main__":
    print("Running UI server in demo mode...")
    print("Open http://localhost:8765/ on your phone or browser.\n")
    asyncio.run(_demo())
