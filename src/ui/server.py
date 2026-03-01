"""
WebSocket + HTTP server for phone UI and demo presenter.

Serves the phone UI (index.html) and demo presenter (demo.html)
over HTTP, pushes real-time updates over WebSocket, and serves
images from a configurable directory.

HTTP routes:
  /              -> phone UI (index.html)
  /demo          -> split-screen demo presenter (demo.html)
  /images/<name> -> serves images from image_dir

WS: ws://host:port/ws -> real-time updates

Message protocol (server -> clients):
  {"type": "step", ...}       -> step update
  {"type": "timer", ...}      -> timer countdown
  {"type": "safety", ...}     -> safety alert
  {"type": "done", ...}       -> recipe complete
  {"type": "image", ...}      -> camera image (for demo presenter)
  {"type": "vlm_result", ...} -> VLM detection output (for demo presenter)

Message protocol (phone -> server):
  {"type": "user_confirm", "step_id": N}
"""

import asyncio
import base64
import json
import mimetypes
from http import HTTPStatus
from pathlib import Path

import websockets


UI_DIR = Path(__file__).parent


class UIServer:
    """WebSocket + HTTP server for phone display and demo presenter."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 image_dir: str | None = None):
        self.host = host
        self.port = port
        self.image_dir = Path(image_dir) if image_dir else None
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

    async def send_image(self, image_path: str, index: int = 0,
                         total: int = 0):
        """Send a camera image to the demo presenter (base64-encoded)."""
        path = Path(image_path)
        if not path.exists():
            return
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
        await self.broadcast({
            "type": "image",
            "base64": image_data,
            "filename": path.name,
            "index": index,
            "total": total,
        })

    async def send_vlm_result(self, result: dict,
                              latency_ms: int = 0):
        """Send VLM detection result to the demo presenter."""
        await self.broadcast({
            "type": "vlm_result",
            "result": result,
            "latency_ms": latency_ms,
        })

    async def process_request(self, path, request_headers):
        """Serve static files for HTTP GET requests."""
        # Phone UI
        if path == "/" or path == "/index.html":
            return self._serve_file(UI_DIR / "index.html",
                                    "text/html; charset=utf-8")

        # Demo presenter
        if path == "/demo" or path == "/demo.html":
            return self._serve_file(UI_DIR / "demo.html",
                                    "text/html; charset=utf-8")

        # Serve images from image_dir
        if path.startswith("/images/") and self.image_dir:
            filename = path[len("/images/"):]
            # Sanitize: only allow simple filenames, no path traversal
            if "/" not in filename and ".." not in filename:
                image_path = self.image_dir / filename
                if image_path.exists():
                    content_type = (
                        mimetypes.guess_type(str(image_path))[0]
                        or "application/octet-stream"
                    )
                    return self._serve_file(image_path, content_type)

        # WebSocket upgrade — let websockets handle it
        if path == "/ws":
            return None

        return None

    def _serve_file(self, file_path: Path, content_type: str):
        """Serve a file as an HTTP response."""
        if not file_path.exists():
            return (HTTPStatus.NOT_FOUND, [], b"Not Found")
        body = file_path.read_bytes()
        return (
            HTTPStatus.OK,
            [("Content-Type", content_type),
             ("Content-Length", str(len(body))),
             ("Cache-Control", "no-cache")],
            body,
        )

    async def start(self):
        """Start the combined HTTP + WebSocket server."""
        print(f"UI server starting:")
        print(f"  Phone UI:    http://{self.host}:{self.port}/")
        print(f"  Demo view:   http://{self.host}:{self.port}/demo")
        print(f"  WebSocket:   ws://{self.host}:{self.port}/ws")
        if self.image_dir:
            print(f"  Images from: {self.image_dir}")

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            process_request=self.process_request,
        ):
            await asyncio.Future()  # Run forever


async def _demo(image_dir: str | None = None):
    """Demo mode: run server and send sample messages with images."""
    server = UIServer(image_dir=image_dir)

    # Collect image files from demo directory if provided
    images = []
    if image_dir:
        img_dir = Path(image_dir)
        images = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

    async def send_demo_sequence():
        # ── 60-SECOND SPEED RUN ──────────────────────────────────
        # Pacing designed for live presentation (~60s total).
        # Elapsed times in comments assume ~0.5s per VLM delay.
        #
        #  0:00  start
        #  0:03  Step 1 — VLM: dice tofu
        #  0:09  Step 1 — VLM detects completion, auto-advance
        #  0:12  Step 2 — user confirm: mince aromatics
        #  0:20  Step 3 — VLM: heat oil
        #  0:25  Step 3 — VLM detects completion, auto-advance
        #  0:29  Step 4 — timer: doubanjiang (30s countdown)
        #  0:34  SAFETY — boil-over detected!
        #  0:41  Step 5 — timer: simmer (5 min shown)
        #  0:50  Step 6 — VLM: cornstarch thickened
        #  0:57  Done — celebration!
        #  1:00  end
        # ─────────────────────────────────────────────────────────
        import time
        t0 = time.monotonic()

        def elapsed():
            s = time.monotonic() - t0
            return f"[{int(s):02d}s]"

        await asyncio.sleep(3)                                     # 0:03
        print(f"\n{elapsed()} Starting demo sequence...")
        total_imgs = len(images)
        img_idx = 0

        async def next_image(vlm_result=None, latency_ms=0):
            nonlocal img_idx
            if img_idx < total_imgs:
                await server.send_image(
                    str(images[img_idx]), img_idx + 1, total_imgs
                )
                img_idx += 1
            if vlm_result:
                await asyncio.sleep(0.5)
                await server.send_vlm_result(vlm_result, latency_ms)

        # ── Step 1: VLM — dice tofu ──
        await server.send_step(1, 6, "Dice the tofu into 1-inch cubes",
                               "vlm", "mapo_tofu")
        print(f"{elapsed()} Step 1: Dice tofu (VLM)")
        await next_image(
            {"dish": "mapo_tofu", "state": "prep", "safe": True,
             "reason": "Tofu on cutting board", "step_complete": False},
            latency_ms=3200
        )
        await asyncio.sleep(5)                                     # 0:09
        await next_image(
            {"dish": "mapo_tofu", "state": "prep", "safe": True,
             "reason": "Diced tofu visible", "step_complete": True},
            latency_ms=2800
        )
        await asyncio.sleep(3)                                     # 0:12

        # ── Step 2: User confirm — mince aromatics ──
        await server.send_step(2, 6, "Mince garlic, ginger, and scallions",
                               "user_confirm", "mapo_tofu")
        print(f"{elapsed()} Step 2: Mince aromatics (user confirm)")
        await next_image(
            {"dish": "mapo_tofu", "state": "prep", "safe": True,
             "reason": "Aromatics being minced", "step_complete": False},
            latency_ms=3100
        )
        await asyncio.sleep(8)                                     # 0:21

        # ── Step 3: VLM — heat oil ──
        await server.send_step(3, 6, "Heat oil in wok until shimmering",
                               "vlm", "mapo_tofu")
        print(f"{elapsed()} Step 3: Heat oil (VLM)")
        await next_image(
            {"dish": "mapo_tofu", "state": "cold", "safe": True,
             "reason": "Oil not hot yet", "step_complete": False},
            latency_ms=4100
        )
        await asyncio.sleep(4.5)                                   # 0:25
        await next_image(
            {"dish": "mapo_tofu", "state": "simmering", "safe": True,
             "reason": "Oil shimmering in wok", "step_complete": True},
            latency_ms=3500
        )
        await asyncio.sleep(3)                                     # 0:29

        # ── Step 4: Timer — doubanjiang ──
        await server.send_step(4, 6, "Add doubanjiang and cook for 30 seconds",
                               "timer", "mapo_tofu")
        print(f"{elapsed()} Step 4: Doubanjiang (timer)")
        await next_image(
            {"dish": "mapo_tofu", "state": "simmering", "safe": True,
             "reason": "Doubanjiang frying", "step_complete": False},
            latency_ms=3300
        )
        await asyncio.sleep(1)
        await server.send_timer("step_4_timer", 30, 30)
        await asyncio.sleep(3.5)                                   # 0:34

        # ── SAFETY ALERT — boil-over! ──
        await next_image(
            {"dish": "mapo_tofu", "state": "boil_over", "safe": False,
             "reason": "Liquid spilling over rim", "step_complete": False},
            latency_ms=2900
        )
        await asyncio.sleep(1)
        await server.send_safety(
            "Boil-over detected! Reduce heat immediately!")
        print(f"{elapsed()} ⚠ SAFETY ALERT: Boil-over!")
        await asyncio.sleep(5.5)                                   # 0:41

        # ── Step 5: Timer — simmer ──
        await server.send_step(5, 6,
                               "Add tofu and broth, simmer for 5 minutes",
                               "timer", "mapo_tofu")
        print(f"{elapsed()} Step 5: Simmer (timer)")
        await next_image(
            {"dish": "mapo_tofu", "state": "simmering", "safe": True,
             "reason": "Tofu simmering in broth", "step_complete": False},
            latency_ms=3800
        )
        await asyncio.sleep(1)
        await server.send_timer("step_5_timer", 15, 300)
        await asyncio.sleep(7)                                     # 0:50

        # ── Step 6: VLM — cornstarch ──
        await server.send_step(6, 6,
                               "Add cornstarch slurry and stir until thickened",
                               "vlm", "mapo_tofu")
        print(f"{elapsed()} Step 6: Cornstarch (VLM)")
        await next_image(
            {"dish": "mapo_tofu", "state": "done", "safe": True,
             "reason": "Thick glossy sauce", "step_complete": True},
            latency_ms=3600
        )
        await asyncio.sleep(6)                                     # 0:57

        # ── Done! ──
        await server.send_done("mapo_tofu")
        print(f"{elapsed()} Done! Mapo tofu complete.")
        await asyncio.sleep(3)                                     # 1:00
        print(f"{elapsed()} Demo sequence finished.")

    await asyncio.gather(
        server.start(),
        send_demo_sequence(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run UI server in demo mode")
    parser.add_argument(
        "--images", default=None,
        help="Directory of demo images to display on right panel")
    args = parser.parse_args()

    print("Running UI server in demo mode...")
    print("Open in browser:")
    print("  Phone only: http://localhost:8765/")
    print("  Full demo:  http://localhost:8765/demo")
    print()
    asyncio.run(_demo(image_dir=args.images))
