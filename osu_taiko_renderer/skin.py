"""Load osu!catch skin elements (Night05 by default) following VRender's
uskin -> default -> skip philosophy.

Catch fruit/drop/banana sprites ship greyscale and are tinted by the combo
colour at draw time (the GL shader multiplies texture * sprite.color); their
`-overlay` companions are drawn untinted on top. The catcher is full-colour
and drawn as-is. @2x variants are preferred when present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# fruit shape cycles by combo index, matching lazer's VisualRepresentation
FRUIT_SHAPES = ["fruit-pear", "fruit-grapes", "fruit-apple", "fruit-orange"]

# elements we load (base name -> whether it also has an -overlay)
_ELEMENTS = {
    "fruit-pear": True, "fruit-grapes": True, "fruit-apple": True, "fruit-orange": True,
    "fruit-drop": True, "fruit-bananas": True,
    "fruit-catcher-idle": False, "fruit-catcher-kiai": False, "fruit-catcher-fail": False,
}

_DEFAULT_COMBOS = [
    (1.0, 0.4, 1.0), (0.4, 0.66, 1.0), (0.4, 0.5, 1.0), (0.62, 0.4, 1.0),
]


class CatchSkin:
    def __init__(self, skin_dir: Path, default_dir: Path | None = None):
        skin_dir = self._resolve_root(skin_dir) if skin_dir else skin_dir
        if default_dir:
            default_dir = self._resolve_root(default_dir)
        self.dirs = [d for d in (skin_dir, default_dir) if d and d.is_dir()]
        self.textures: dict[str, np.ndarray] = {}
        self._load_elements()
        self.combo_colors = self._load_combos()
        self.catcher_aspect = self._aspect("fruit-catcher-idle", 324 / 305)

    @staticmethod
    def _resolve_root(d: Path) -> Path:
        """Find the actual skin root. The bundled default extracts its sprites
        into a `_default-source` subdir (danser appends it manually), and some
        .osk archives wrap everything in one folder — so if the given dir has
        no skin.ini / catch sprites, descend one level to a subdir that does."""
        d = Path(d)

        def has_skin(p: Path) -> bool:
            return (p / "skin.ini").is_file() or any(p.glob("fruit-*.png"))

        if not d.is_dir() or has_skin(d):
            return d
        subs = sorted(c for c in d.iterdir() if c.is_dir())
        for sub in subs:
            if has_skin(sub):
                return sub
        return d

    # --- public ---------------------------------------------------------------

    def fruit_key(self, combo_index: int) -> str:
        return FRUIT_SHAPES[combo_index % len(FRUIT_SHAPES)]

    def combo_color(self, combo_index: int) -> tuple[float, float, float]:
        if not self.combo_colors:
            return (1.0, 1.0, 1.0)
        return self.combo_colors[combo_index % len(self.combo_colors)]

    def has(self, key: str) -> bool:
        return key in self.textures

    # --- loading --------------------------------------------------------------

    def _resolve(self, basename: str) -> Path | None:
        for d in self.dirs:
            for stem in (f"{basename}@2x", basename):
                p = d / f"{stem}.png"
                if p.is_file():
                    return p
        return None

    # fruit/drop/banana bases ship dark-grey in many skins; brighten them so
    # the combo-colour tint reads vibrant (osu draws fruit bright, not muddy).
    _BOOST = {"fruit-pear", "fruit-grapes", "fruit-apple", "fruit-orange",
              "fruit-drop", "fruit-bananas"}

    def _load_elements(self) -> None:
        for base, has_overlay in _ELEMENTS.items():
            keys = [base] + ([f"{base}-overlay"] if has_overlay else [])
            for key in keys:
                p = self._resolve(key)
                if p is None:
                    continue
                tex = _rgba(p)
                if key in self._BOOST:
                    tex = _brighten(tex, 1.9)
                self.textures[key] = tex

    def _load_combos(self) -> list[tuple[float, float, float]]:
        for d in self.dirs:
            ini = d / "skin.ini"
            if not ini.is_file():
                continue
            combos: dict[int, tuple[float, float, float]] = {}
            for line in ini.read_text(errors="replace").splitlines():
                s = line.strip()
                low = s.lower()
                if low.startswith("combo") and ":" in s and low[5:6].isdigit():
                    idx_str, val = s.split(":", 1)
                    try:
                        idx = int(idx_str.strip()[5:])
                        r, g, b = (int(x) for x in val.split(",")[:3])
                        combos[idx] = (r / 255, g / 255, b / 255)
                    except ValueError:
                        continue
            if combos:
                return [combos[k] for k in sorted(combos)]
        return list(_DEFAULT_COMBOS)

    def _aspect(self, key: str, default: float) -> float:
        t = self.textures.get(key)
        if t is None:
            return default
        h, w = t.shape[:2]
        return h / w if w else default


def _rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def _brighten(rgba: np.ndarray, factor: float) -> np.ndarray:
    out = rgba.astype("f4")
    out[..., :3] = np.clip(out[..., :3] * factor, 0, 255)
    return out.astype("u1")
