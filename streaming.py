"""
streaming.py — utilities for handling `stream=true` chat completions.

Two distinct problems solved here:

1. **Cost accounting on a stream.** We don't know completion tokens until
   the stream ends. `StreamCollector` is a mutable sink that provider
   streaming methods write into (prompt/completion tokens if the upstream
   reports them, or the accumulated text otherwise) so the caller can
   compute actual cost and record usage once the generator is exhausted.

2. **Unmasking across chunk boundaries.** If a redacted placeholder like
   `[EMAIL_1]` gets split across two SSE chunks (e.g. "...[EMAIL" then
   "_1]..."), a naive per-chunk string replace would silently fail to
   unmask it. `StreamUnmasker` buffers only the minimum necessary — text
   from the last unmatched "[" onward — and releases everything before
   that immediately, so streaming stays effectively real-time while still
   catching placeholders that span chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamCollector:
    """Accumulates stats across a streamed response for post-hoc cost
    accounting and cache population."""

    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    response_id: str = ""


class StreamUnmasker:
    """Incrementally replaces placeholder tokens in a stream of text pieces.

    Usage:
        unmasker = StreamUnmasker(mapping)
        for piece in incoming_pieces:
            safe_text = unmasker.feed(piece)   # may be "" if buffering
            ...
        leftover = unmasker.flush()            # call once at stream end
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self._buffer = ""

    def _replace_known_placeholders(self, text: str) -> str:
        for placeholder in sorted(self._mapping, key=len, reverse=True):
            if placeholder in text:
                text = text.replace(placeholder, self._mapping[placeholder])
        return text

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        self._buffer += piece

        last_open = self._buffer.rfind("[")
        last_close = self._buffer.rfind("]")

        if last_open == -1 or last_close > last_open:
            # No unresolved "[" — everything is safe to emit.
            safe, self._buffer = self._buffer, ""
        else:
            # Hold back from the last unmatched "[" onward; it might be the
            # start of a placeholder that continues in the next chunk.
            safe, self._buffer = self._buffer[:last_open], self._buffer[last_open:]

        return self._replace_known_placeholders(safe)

    def flush(self) -> str:
        """Call at stream end to release any buffered tail (e.g. a
        trailing literal '[' that never turned out to be a placeholder)."""
        remainder = self._replace_known_placeholders(self._buffer)
        self._buffer = ""
        return remainder
