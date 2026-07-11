"""Argon counter font (the segmented LED-look digits) — ArgonCounterTextComponent.

Each glyph is an osu-resources texture (Gameplay/Fonts/argon-counter-*, 240px);
numbers render as a dim 'wireframes' ghost layer with the bright digit on top,
fixed-width. Glyphs bundled under glyphs/ (from the mania renderer's extraction).

Compositing note: the glyph + wireframe sprites are pure WHITE with the shape
carried entirely in the ALPHA channel (catch/STD force this too). So the counter
composites in the ALPHA domain only and keeps RGB flat-white — the lit digit is
laid *over* the dim wireframe as `a_digit + a_wire*(1-a_digit)`. This mirrors
catch's argon_counter/argon_hud (flat white RGB, alpha = coverage) and avoids the
premultiply-into-a-transparent-buffer darkening that used to grey-out the thin
antialiased segment edges at small HUD cell sizes (score/combo/accuracy → tofu).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_DIR = Path(__file__).resolve().parent / "glyphs"
_NAME = {**{str(i): f"argon_{i}" for i in range(10)},
         ".": "argon_dot", "%": "argon_percent", "x": "argon_x"}
_NATIVE = 240


def _over(dst_a: np.ndarray, src_a: np.ndarray, x: int) -> None:
    """Source-over composite the alpha patch `src_a` onto `dst_a` at column x.
    Both are single-channel float alpha in [0,1]; RGB is handled separately
    (flat white) so only coverage combines: out = src + dst*(1-src)."""
    H, W = dst_a.shape
    h, w = src_a.shape
    x0, y0 = max(0, x), 0
    x1, y1 = min(W, x + w), min(H, h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src_a[y0:y1, x0 - x:x1 - x]
    d = dst_a[y0:y1, x0:x1]
    dst_a[y0:y1, x0:x1] = s + d * (1.0 - s)


class ArgonCounter:
    def __init__(self):
        self.g: dict[str, np.ndarray] = {}
        for ch, name in _NAME.items():
            p = _DIR / f"{name}.png"
            self.g[ch] = np.array(Image.open(p).convert("RGBA")).astype(np.float32) / 255.0
        self.wire = np.array(Image.open(_DIR / "argon_wireframes.png")
                             .convert("RGBA")).astype(np.float32) / 255.0
        self._rcache: dict = {}      # (text, cell, wire_alpha, color) -> rgba
        self._scache: dict = {}      # (char|'wire', cell) -> scaled alpha field

    def _scaled_alpha(self, ch, arr, cell):
        """The glyph's ALPHA channel resized to the cell (cached)."""
        key = (ch, cell)
        hit = self._scache.get(key)
        if hit is not None:
            return hit
        a = arr[..., 3]
        out = (np.asarray(Image.fromarray((a * 255).astype(np.uint8))
               .resize((cell, cell), Image.LANCZOS)).astype(np.float32) / 255.0)
        self._scache[key] = out
        return out

    def measure(self, text, digit_h):
        s = digit_h / _NATIVE
        cell = max(1, int(round(_NATIVE * s)))
        gap = int(round(cell * 0.03))
        return len(text) * cell - (len(text) - 1) * gap, cell

    def render(self, text: str, digit_h: float, *, wire_alpha=0.25,
               color=(255, 255, 255)) -> np.ndarray:
        cell = max(1, int(round(_NATIVE * (digit_h / _NATIVE))))
        key = (text, cell, round(wire_alpha, 2), color)
        hit = self._rcache.get(key)
        if hit is not None:
            return hit
        gap = int(round(cell * 0.03))                 # slight overlap (spacing -2)
        W = max(1, len(text) * cell - (len(text) - 1) * gap)
        # Accumulate ALPHA only; RGB is flat white (all sprites are white) so
        # antialiased digit edges never get darkened toward the ghost.
        alpha = np.zeros((cell, W), np.float32)
        wi = self._scaled_alpha("wire", self.wire, cell)
        x = 0
        for ch in text:
            g = self.g.get(ch)
            if g is not None:
                if ch.isdigit():
                    _over(alpha, wi * wire_alpha, x)          # dim wireframe ghost
                _over(alpha, self._scaled_alpha(ch, g, cell), x)  # lit glyph over it
            x += cell - gap
        out8 = np.zeros((cell, W, 4), np.uint8)
        out8[..., 0] = color[0]
        out8[..., 1] = color[1]
        out8[..., 2] = color[2]
        out8[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        if len(self._rcache) > 2048:
            self._rcache.clear()
        self._rcache[key] = out8
        return out8
