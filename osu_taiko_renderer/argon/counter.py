"""Argon counter font (the segmented LED-look digits) — ArgonCounterTextComponent.

Each glyph is an osu-resources texture (Gameplay/Fonts/argon-counter-*, 240px);
numbers render as a dim 'wireframes' ghost layer with the bright digit on top,
fixed-width. Glyphs bundled under glyphs/ (from the mania renderer's extraction).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_DIR = Path(__file__).resolve().parent / "glyphs"
_NAME = {**{str(i): f"argon_{i}" for i in range(10)},
         ".": "argon_dot", "%": "argon_percent", "x": "argon_x"}
_NATIVE = 240


def _paste(dst, src, x, y, alpha=1.0):
    """Straight-alpha composite RGBA float src into RGBA float dst at (x,y)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4] * alpha
    region = dst[y0:y1, x0:x1]
    region[..., :3] = region[..., :3] * (1 - a) + s[..., :3] * a
    region[..., 3:4] = region[..., 3:4] + a * (1 - region[..., 3:4])


class ArgonCounter:
    def __init__(self):
        self.g: dict[str, np.ndarray] = {}
        for ch, name in _NAME.items():
            p = _DIR / f"{name}.png"
            self.g[ch] = np.array(Image.open(p).convert("RGBA")).astype(np.float32) / 255.0
        self.wire = np.array(Image.open(_DIR / "argon_wireframes.png")
                             .convert("RGBA")).astype(np.float32) / 255.0
        self._rcache: dict = {}      # (text, cell, wire_alpha, color) -> rgba
        self._scache: dict = {}      # (char|'wire', cell) -> scaled float glyph

    def _scaled(self, ch, arr, cell):
        key = (ch, cell)
        hit = self._scache.get(key)
        if hit is not None:
            return hit
        out = (np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
               .resize((cell, cell), Image.LANCZOS)).astype(np.float32) / 255.0)
        self._scache[key] = out
        return out

    def measure(self, text, digit_h):
        s = digit_h / _NATIVE
        cell = max(1, int(round(_NATIVE * s)))
        gap = int(round(cell * 0.03))
        return len(text) * cell - (len(text) - 1) * gap, cell

    def render(self, text: str, digit_h: float, *, wire_alpha=0.33,
               color=(255, 255, 255)) -> np.ndarray:
        cell = max(1, int(round(_NATIVE * (digit_h / _NATIVE))))
        key = (text, cell, round(wire_alpha, 2), color)
        hit = self._rcache.get(key)
        if hit is not None:
            return hit
        gap = int(round(cell * 0.03))                 # slight overlap (spacing -2)
        W = max(1, len(text) * cell - (len(text) - 1) * gap)
        out = np.zeros((cell, W, 4), np.float32)
        wi = self._scaled("wire", self.wire, cell)
        x = 0
        for ch in text:
            g = self.g.get(ch)
            if g is not None:
                if ch.isdigit():
                    _paste(out, wi, x, 0, alpha=wire_alpha)
                _paste(out, self._scaled(ch, g, cell), x, 0)
            x += cell - gap
        out8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        if color != (255, 255, 255):
            tint = np.array(color, np.float32) / 255.0
            out8[..., :3] = (out8[..., :3].astype(np.float32) * tint).astype(np.uint8)
        if len(self._rcache) > 2048:
            self._rcache.clear()
        self._rcache[key] = out8
        return out8
