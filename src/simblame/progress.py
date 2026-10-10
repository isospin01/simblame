"""Progress bars and status lines on stderr (no dependencies, quiet when not a terminal)."""

from __future__ import annotations

import contextlib
import sys
import time


class Progress:
    """``with progress.task("v1.1", total=50) as t: ... t.advance()``.

    On a terminal this draws a live bar with an ETA; otherwise it stays quiet
    (warnings are always printed).
    """

    def __init__(self, enabled: bool = True, stream=None):
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self.live = enabled and getattr(self.stream, "isatty", lambda: False)()
        self._width = 0

    @contextlib.contextmanager
    def task(self, label: str, total: int):
        task = _Task(self, label, total)
        try:
            yield task
        finally:
            self._clear()

    def status(self, message: str) -> None:
        """A one-line note about what is happening (e.g. 'creating env for mujoco 3.4.0')."""
        if self.enabled:
            self._clear()
            self.stream.write(f"  {message}\n")
            self.stream.flush()

    def warn(self, message: str) -> None:
        self._clear()
        self.stream.write(f"warning: {message}\n")
        self.stream.flush()

    def _draw(self, text: str) -> None:
        if not self.live:
            return
        pad = " " * max(0, self._width - len(text))
        self.stream.write("\r" + text + pad)
        self.stream.flush()
        self._width = len(text)

    def _clear(self) -> None:
        if self.live and self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
            self._width = 0


class _Task:
    BAR = 24

    def __init__(self, progress: Progress, label: str, total: int):
        self.p, self.label, self.total = progress, label, max(total, 1)
        self.done = 0
        self.start = time.time()
        self._render("")

    def advance(self, detail: str = "") -> None:
        self.done += 1
        self._render(detail)

    def _render(self, detail: str) -> None:
        frac = min(self.done / self.total, 1.0)
        filled = int(round(frac * self.BAR))
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0.0
        label = self.label if len(self.label) <= 32 else self.label[:29] + "..."
        text = (f"  {label:<32} [{'#' * filled}{'.' * (self.BAR - filled)}] "
                f"{self.done}/{self.total}  {_fmt_time(elapsed)} elapsed"
                + (f", ~{_fmt_time(eta)} left" if self.done and self.done < self.total else "")
                + (f"  {detail}" if detail else ""))
        self.p._draw(text)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds // 60:.0f}m{seconds % 60:02.0f}s"
