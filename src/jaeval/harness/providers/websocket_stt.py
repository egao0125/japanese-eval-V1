"""WebSocket-based STT provider for remote inference servers.

Connects to a WebSocket STT server (e.g., the voice-fullduplex
Qwen3 inference server on RunPod) and sends audio for transcription.

Protocol:
    1. Connect to ws://host:port
    2. Send raw WAV bytes as a binary message
    3. Receive JSON response: {"text": "...", "latency": ...}
    4. Close connection

The server-side latency (if provided) is recorded in metadata,
while the client-measured round-trip latency is used for RTF
calculations.
"""

from __future__ import annotations

import asyncio
import json
import time

from .base import STTProvider, TranscribeResult


class WebSocketSTTProvider(STTProvider):
    """WebSocket STT provider for remote inference servers.

    Args:
        url: WebSocket URL of the STT server (e.g., ``ws://localhost:8766``).
        timeout_sec: Connection and message timeout in seconds.
    """

    name = "websocket"
    requires_gpu = False

    def __init__(self, *, url: str = "ws://localhost:8766", timeout_sec: float = 30.0):
        self.url = url
        self.timeout_sec = timeout_sec

    async def _transcribe_async(self, audio_bytes: bytes) -> tuple[str, float, dict]:
        """Send audio over WebSocket and receive transcript.

        Returns:
            Tuple of (transcript, round_trip_latency, server_metadata).
        """
        import websockets

        t0 = time.monotonic()

        async with websockets.connect(
            self.url,
            open_timeout=self.timeout_sec,
            close_timeout=self.timeout_sec,
        ) as ws:
            await ws.send(audio_bytes)
            response_raw = await asyncio.wait_for(
                ws.recv(), timeout=self.timeout_sec
            )

        latency = time.monotonic() - t0

        # Parse response -- expect JSON with at least "text" field
        if isinstance(response_raw, bytes):
            response_raw = response_raw.decode("utf-8")

        try:
            response = json.loads(response_raw)
            transcript = response.get("text", "")
            server_meta = {
                k: v for k, v in response.items() if k != "text"
            }
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat raw response as plain text transcript
            transcript = str(response_raw).strip()
            server_meta = {}

        return transcript, latency, server_meta

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 8000) -> TranscribeResult:
        """Transcribe audio by sending it to a remote WebSocket STT server.

        Args:
            audio_bytes: WAV file bytes (including header).
            sample_rate: Sample rate of the audio in Hz (informational;
                the server determines format from the WAV header).

        Returns:
            TranscribeResult with transcript and round-trip latency.
        """
        # Run the async WebSocket call in a sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an existing event loop -- create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                transcript, latency, server_meta = pool.submit(
                    lambda: asyncio.run(self._transcribe_async(audio_bytes))
                ).result(timeout=self.timeout_sec + 5)
        else:
            transcript, latency, server_meta = asyncio.run(
                self._transcribe_async(audio_bytes)
            )

        return TranscribeResult(
            text=transcript,
            latency_sec=latency,
            metadata={"url": self.url, **server_meta},
        )
