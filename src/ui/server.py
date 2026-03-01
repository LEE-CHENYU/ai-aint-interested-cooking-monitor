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
from pathlib import Path

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response


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
        self._loop = None

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
        """Send a camera image file to the demo presenter (base64-encoded)."""
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

    async def send_image_frame(self, frame, filename: str = "",
                               index: int = 0, total: int = 0):
        """Send a camera frame (numpy array) as base64 PNG to the demo UI."""
        import cv2
        _, buf = cv2.imencode('.png', frame)
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        await self.broadcast({
            "type": "image",
            "base64": b64,
            "filename": filename,
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

    async def process_request(self, connection, request):
        """Serve static files for HTTP GET requests (websockets 13+ API)."""
        # Phone UI
        if request.path in ("/", "/index.html"):
            return self._serve_file(UI_DIR / "index.html",
                                    "text/html; charset=utf-8")
        # Demo presenter
        if request.path in ("/demo", "/demo.html"):
            return self._serve_file(UI_DIR / "demo.html",
                                    "text/html; charset=utf-8")
        # Serve images from image_dir
        if request.path.startswith("/images/") and self.image_dir:
            filename = request.path[len("/images/"):]
            if "/" not in filename and ".." not in filename:
                image_path = self.image_dir / filename
                if image_path.exists():
                    content_type = (
                        mimetypes.guess_type(str(image_path))[0]
                        or "application/octet-stream"
                    )
                    return self._serve_file(image_path, content_type)
        # Allow WebSocket upgrade
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        # Reject non-WebSocket requests to unknown paths (e.g. /favicon.ico)
        return Response(
            404,
            "Not Found",
            Headers([("Content-Length", "0")]),
            b"",
        )

    @staticmethod
    def _serve_file(file_path: Path, content_type: str = "text/html; charset=utf-8"):
        """Return an HTTP Response for a file."""
        if not file_path.exists():
            return Response(
                404, "Not Found",
                Headers([("Content-Length", "0")]),
                b"",
            )
        body = file_path.read_bytes()
        return Response(
            200,
            "OK",
            Headers([
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-cache"),
            ]),
            body,
        )

    # ------------------------------------------------------------------
    # Thread-safe sync wrappers (called from agent thread)
    # ------------------------------------------------------------------

    def fire_step(self, step_id: int, total_steps: int,
                  instruction: str, completion_type: str,
                  dish: str = ""):
        """Schedule send_step() from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_step(step_id, total_steps, instruction,
                               completion_type, dish),
                self._loop,
            )

    def fire_timer(self, name: str, remaining_seconds: int,
                   total_seconds: int):
        """Schedule send_timer() from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_timer(name, remaining_seconds, total_seconds),
                self._loop,
            )

    def fire_safety(self, message: str, severity: str = "critical"):
        """Schedule send_safety() from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_safety(message, severity),
                self._loop,
            )

    def fire_done(self, dish: str):
        """Schedule send_done() from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_done(dish),
                self._loop,
            )

    def fire_image(self, frame, filename: str = "",
                   index: int = 0, total: int = 0):
        """Schedule send_image_frame() from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_image_frame(frame, filename, index, total),
                self._loop,
            )

    def fire_vlm_result(self, result: dict, latency_ms: int = 0):
        """Schedule send_vlm_result() from any thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_vlm_result(result, latency_ms),
                self._loop,
            )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the combined HTTP + WebSocket server."""
        self._loop = asyncio.get_running_loop()

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


# ── TIMING PROFILES ─────────────────────────────────────────────
# Each profile is a dict of sleep durations (seconds) keyed by
# event name.  The demo sequence uses the same events regardless
# of duration — only the pacing changes.
#
# 60s   ≈  1-minute speed run
# 120s  ≈  2-minute demo (comfortable narration, light timer ticks)
# 5m    ≈  5-minute full presentation (deep narration, live countdowns)

TIMING = {
    "60s": {
        "startup":       3,
        "s1_img1":       5,     "s1_img2":       3,
        "s2":            8,
        "s3_img1":       4.5,   "s3_img2":       3,
        "s4_img":        1,     "s4_timer":      3.5,
        "safety_img":    1,     "safety_alert":  5.5,
        "s5_img":        1,     "s5_timer":      7,
        "s6":            6,
        "done":          3,
        "s4_ticks":      0,
        "s5_ticks":      0,
    },
    "120s": {
        "startup":       3,
        "s1_img1":       10,    "s1_img2":       6,
        "s2":            18,
        "s3_img1":       10,    "s3_img2":       6,
        "s4_img":        2,     "s4_timer":      5,
        "safety_img":    2,     "safety_alert":  12,
        "s5_img":        2,     "s5_timer":      14,
        "s6":            10,
        "done":          5,
        "s4_ticks":      5,     # 5s of visible countdown (30→25)
        "s5_ticks":      5,     # 5s of visible countdown (300→295)
    },
    "5m": {
        "startup":       5,
        "s1_img1":       30,    "s1_img2":       15,
        "s2":            40,
        "s3_img1":       25,    "s3_img2":       15,
        "s4_img":        5,     "s4_timer":      5,
        "safety_img":    5,     "safety_alert":  25,
        "s5_img":        5,     "s5_timer":      8,
        "s6":            30,
        "done":          22,
        "s4_ticks":      20,
        "s5_ticks":      40,
    },
}


async def _demo(image_dir: str | None = None,
                duration: str = "60s"):
    """Demo mode: run server and send sample messages with images."""
    server = UIServer(image_dir=image_dir)
    t = TIMING[duration]

    # Collect image files from demo directory if provided
    images = []
    if image_dir:
        img_dir = Path(image_dir)
        images = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

    async def send_demo_sequence():
        import time
        t0 = time.monotonic()

        def elapsed():
            s = time.monotonic() - t0
            m, sec = divmod(int(s), 60)
            return f"[{m}:{sec:02d}]"

        await asyncio.sleep(t["startup"])
        print(f"\n{elapsed()} Starting {duration} demo sequence...")
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

        async def tick_timer(name, start, total, ticks):
            """Send live countdown updates so the phone timer visibly ticks."""
            for i in range(ticks):
                remaining = start - i - 1
                if remaining < 0:
                    break
                await asyncio.sleep(1)
                await server.send_timer(name, remaining, total)

        # ── Step 1: VLM — rinse rice ──
        await server.send_step(1, 6,
                               "Rinse rice until water runs clear",
                               "vlm", "rice")
        print(f"{elapsed()} Step 1: Rinse rice (VLM)")
        await next_image(
            {"dish": "rice", "state": "cold", "safe": True,
             "reason": "Dry rice in pot", "step_complete": False},
            latency_ms=3200
        )
        await asyncio.sleep(t["s1_img1"])
        await next_image(
            {"dish": "rice", "state": "cold", "safe": True,
             "reason": "Rice rinsed, water clear", "step_complete": True},
            latency_ms=2800
        )
        await asyncio.sleep(t["s1_img2"])

        # ── Step 2: User confirm — add water ──
        await server.send_step(2, 6,
                               "Add water at 1:1.5 rice-to-water ratio",
                               "user_confirm", "rice")
        print(f"{elapsed()} Step 2: Add water (user confirm)")
        await next_image(
            {"dish": "rice", "state": "cold", "safe": True,
             "reason": "Water added to pot", "step_complete": False},
            latency_ms=3100
        )
        await asyncio.sleep(t["s2"])

        # ── Step 3: VLM — bring to boil ──
        await server.send_step(3, 6,
                               "Bring water to a rolling boil on high heat",
                               "vlm", "rice")
        print(f"{elapsed()} Step 3: Bring to boil (VLM)")
        await next_image(
            {"dish": "rice", "state": "cold", "safe": True,
             "reason": "Water not boiling yet", "step_complete": False},
            latency_ms=4100
        )
        await asyncio.sleep(t["s3_img1"])
        await next_image(
            {"dish": "rice", "state": "boiling", "safe": True,
             "reason": "Rolling boil detected", "step_complete": True},
            latency_ms=3500
        )
        await asyncio.sleep(t["s3_img2"])

        # ── Step 4: Timer — simmer 15 min ──
        await server.send_step(4, 6,
                               "Reduce heat, cover and simmer for 15 minutes",
                               "timer", "rice")
        print(f"{elapsed()} Step 4: Simmer (timer)")
        await next_image(
            {"dish": "rice", "state": "simmering", "safe": True,
             "reason": "Gentle simmer, covered", "step_complete": False},
            latency_ms=3300
        )
        await asyncio.sleep(t["s4_img"])
        await server.send_timer("step_4_timer", 900, 900)
        if t["s4_ticks"]:
            await tick_timer("step_4_timer", 900, 900, t["s4_ticks"])
        await asyncio.sleep(t["s4_timer"])

        # ── SAFETY ALERT — boil-over! ──
        await next_image(
            {"dish": "rice", "state": "boil_over", "safe": False,
             "reason": "Starchy water spilling over rim",
             "step_complete": False},
            latency_ms=2900
        )
        await asyncio.sleep(t["safety_img"])
        await server.send_safety(
            "Boil-over detected! Reduce heat immediately!")
        print(f"{elapsed()} SAFETY ALERT: Boil-over!")
        await asyncio.sleep(t["safety_alert"])

        # ── Step 5: Timer — let stand 5 min ──
        await server.send_step(5, 6,
                               "Remove from heat, let stand covered 5 minutes",
                               "timer", "rice")
        print(f"{elapsed()} Step 5: Let stand (timer)")
        await next_image(
            {"dish": "rice", "state": "simmering", "safe": True,
             "reason": "Rice resting, lid on", "step_complete": False},
            latency_ms=3800
        )
        await asyncio.sleep(t["s5_img"])
        await server.send_timer("step_5_timer", 300, 300)
        if t["s5_ticks"]:
            await tick_timer("step_5_timer", 300, 300, t["s5_ticks"])
        await asyncio.sleep(t["s5_timer"])

        # ── Step 6: VLM — fluff and serve ──
        await server.send_step(6, 6,
                               "Fluff rice with a fork and serve",
                               "vlm", "rice")
        print(f"{elapsed()} Step 6: Fluff and serve (VLM)")
        await next_image(
            {"dish": "rice", "state": "done", "safe": True,
             "reason": "Fluffy rice, ready to serve",
             "step_complete": True},
            latency_ms=3600
        )
        await asyncio.sleep(t["s6"])

        # ── Done! ──
        await server.send_done("rice")
        print(f"{elapsed()} Done! Rice is ready.")
        await asyncio.sleep(t["done"])
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
    parser.add_argument(
        "--duration", choices=["60s", "120s", "5m"], default="60s",
        help="Demo pacing: 60s speed run, 120s standard, or 5m full")
    args = parser.parse_args()

    print(f"Running UI server in demo mode ({args.duration})...")
    print("Open in browser:")
    print("  Phone only: http://localhost:8765/")
    print("  Full demo:  http://localhost:8765/demo")
    print()
    asyncio.run(_demo(image_dir=args.images, duration=args.duration))
