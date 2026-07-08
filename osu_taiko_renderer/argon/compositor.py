"""Additive effects pass over the readback frame: hit explosions and floating
GREAT/OK/MISS judgement text — both additive-blended like osu!lazer Argon.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import _const as C
from .font import get_font
from .textures import bake_drum_flash, bake_explosion, bake_ring

_JUDGE_TEXT = {"great": "GREAT", "ok": "OK", "miss": "MISS"}
_JUDGE_COL = {"great": C.JUDGE_GREAT, "ok": C.JUDGE_OK, "miss": C.JUDGE_MISS}


def _add_tex(rgb, tex, cx, cy, tw, th, intensity, tint=(1.0, 1.0, 1.0)):
    """Additive-blend RGBA `tex` (resized to tw×th) centred at (cx,cy)."""
    tw, th = int(round(tw)), int(round(th))
    if tw < 1 or th < 1 or intensity <= 0:
        return
    # BILINEAR (not LANCZOS): this runs per active popup per frame and the
    # textures are soft glows/text — the quality difference is invisible, the
    # speedup is ~3x. (One-time bakes stay LANCZOS.)
    im = np.asarray(Image.fromarray(tex).resize((tw, th), Image.BILINEAR)).astype(np.float32)
    src_rgb, src_a = im[..., :3], (im[..., 3:4] / 255.0) * intensity
    if tint != (1.0, 1.0, 1.0):
        src_rgb = src_rgb * np.asarray(tint, dtype=np.float32)
    H, W = rgb.shape[:2]
    x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + tw), min(H, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = src_a[sy0:sy0 + h, sx0:sx0 + w]
    sc = src_rgb[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1].astype(np.float32) + sc * sa
    rgb[dy0:dy1, dx0:dx1] = np.clip(region, 0, 255).astype(np.uint8)


_BRIGHT_LUT = None


def bloom(rgb, *, thresh=130, strength=0.55, step=6):
    """Cheap additive bloom (approximates osu!lazer's glow). All work is done in
    C via PIL: downscale → bright-pass (point LUT) → gaussian-blur the small
    image → upscale → saturating add. Avoids per-frame numpy full-frame floats."""
    global _BRIGHT_LUT
    if _BRIGHT_LUT is None:
        _BRIGHT_LUT = [int(max(0, v - thresh) * strength) for v in range(256)]
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    sw, sh = max(1, W // step), max(1, H // step)
    small = img.resize((sw, sh), Image.BILINEAR).point(_BRIGHT_LUT * 3)
    small = small.filter(ImageFilter.GaussianBlur(radius=max(2, sw * 0.02)))
    up = small.resize((W, H), Image.BILINEAR)
    return np.array(ImageChops.add(img, up))


def _blit_straight(rgb, tex, cx, cy, tw, th, alpha):
    """Straight-alpha composite RGBA `tex` (resized to tw×th) centred at (cx,cy),
    scaled by `alpha` (for fades). For skin judgement images (not additive)."""
    tw, th = int(round(tw)), int(round(th))
    if tw < 1 or th < 1 or alpha <= 0:
        return
    im = np.asarray(Image.fromarray(tex).resize((tw, th), Image.BILINEAR)).astype(np.float32)
    src_rgb, src_a = im[..., :3], (im[..., 3:4] / 255.0) * alpha
    H, W = rgb.shape[:2]
    x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + tw), min(H, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = src_a[sy0:sy0 + h, sx0:sx0 + w]
    sc = src_rgb[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1].astype(np.float32)
    region = region * (1 - sa) + sc * sa
    rgb[dy0:dy1, dx0:dx1] = np.clip(region, 0, 255).astype(np.uint8)


def _add_prescaled(rgb, im, cx, cy, intensity):
    """Additive-blend an already-float32 RGBA texture (no resize) centred at (cx,cy)."""
    th, tw = im.shape[:2]
    H, W = rgb.shape[:2]
    x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + tw), min(H, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = (im[sy0:sy0 + h, sx0:sx0 + w, 3:4] / 255.0) * intensity
    sc = im[sy0:sy0 + h, sx0:sx0 + w, :3]
    region = rgb[dy0:dy1, dx0:dx1].astype(np.float32) + sc * sa
    rgb[dy0:dy1, dx0:dx1] = np.clip(region, 0, 255).astype(np.uint8)


class ArgonEffects:
    def __init__(self, geo, skin_dir=None):
        self.geo = geo
        self.exp = {
            False: bake_explosion(C.CENTRE_HIT_GRAD, C.CENTRE_HIT_GLOW),  # centre/don
            True: bake_explosion(C.RIM_HIT_GRAD, C.RIM_HIT_GLOW),         # rim/kat
        }
        self._ring = bake_ring(64, 8)   # white judgement RingExplosion ring
        self.font = get_font("SemiBold")
        self._jcache: dict[str, np.ndarray] = {}
        # Skin judgement images (taiko-hit300/100/0) — used for the gameplay
        # popups instead of Torus text when the skin provides them.
        from ..taiko_skin import TaikoSkin
        skin = TaikoSkin(skin_dir)
        self._skin_judge = {}
        for res, name in (("great", "taiko-hit300"), ("ok", "taiko-hit100"),
                          ("miss", "taiko-hit0")):
            img = skin.load(name)
            if img is not None:
                self._skin_judge[res] = img
        self._use_skin_judge = "great" in self._skin_judge
        # Pre-scale explosion textures to the two note sizes (resizing every
        # frame per active explosion was a render-time hotspot).
        self._exp_scaled = {}
        for is_rim in (False, True):
            for big in (False, True):
                d = int(round(geo.big_d if big else geo.note_d))
                im = np.asarray(Image.fromarray(self.exp[is_rim])
                                .resize((d, d), Image.LANCZOS)).astype(np.float32)
                self._exp_scaled[(is_rim, big)] = im
        # Pre-scale the 4 drum-flash quadrants to the drum size.
        dd = int(round(geo.drum_d))
        self._drum_scaled = {}
        for is_rim in (False, True):
            for left in (True, False):
                im = np.asarray(Image.fromarray(bake_drum_flash(ring=is_rim, left=left))
                                .resize((dd, dd), Image.LANCZOS)).astype(np.float32)
                self._drum_scaled[(is_rim, left)] = im

    def _judge_tex(self, result):
        if result not in self._jcache:
            if self._use_skin_judge and result in self._skin_judge:
                img = self._skin_judge[result]
                th = int(self.geo.note_d * 1.1)
                tw = max(1, int(th * img.shape[1] / img.shape[0]))
                self._jcache[result] = np.array(
                    Image.fromarray(img).resize((tw, th), Image.LANCZOS))
                return self._jcache[result]
            # ArgonJudgementPiece: plain straight-alpha OsuFont text, no glow
            # halo (the only burst is the separate RingExplosion). Just the text.
            px = self.geo.note_d * 0.46
            txt = self.font.render(_JUDGE_TEXT[result], px, color=_JUDGE_COL[result],
                                   spacing=C.JUDGE_SPACING * self.geo.scale)
            self._jcache[result] = txt
        return self._jcache[result]

    _RING_SPEC = {"great": (4, 4, 1.0), "ok": (4, 0, 0.6)}   # (small,large,travel_x); miss none

    def _ring_burst(self, rgb, res, age, rt, g):
        """lazer taiko ArgonJudgementPiece.RingExplosion: white hollow rings
        burst outward from the hit target, tinted by the result colour,
        additive. travel 58, start_position_ratio 0.6, fade 1000ms OutQuint.
        Seeded on the judged time so it is stable across frames."""
        spec = self._RING_SPEC.get(res)
        if spec is None:
            return
        n_small, n_large, tmult = spec
        ga = max(0.0, (1.0 - age / 1000.0)) ** 5
        if ga <= 0.004:
            return
        import math, random
        col = tuple(c / 255.0 for c in _JUDGE_COL[res][:3])
        sc = g.scale
        travel = 58.0 * sc * tmult
        p = min(age, 600) / 600.0
        rad = 0.6 + 0.4 * (1.0 - (1.0 - p) ** 5)
        pieces = [9.0 * sc] * n_small + [14.0 * sc] * n_large
        for i, size in enumerate(pieces):
            rng = random.Random((int(rt) * 1000003) ^ (i * 2654435761))
            d = rng.uniform(0.0, 360.0)
            dist = rng.uniform(travel / 2.0, travel)
            cur = dist * rad
            _add_tex(rgb, self._ring, g.target_x + math.cos(d) * cur,
                     g.center_y + math.sin(d) * cur, size, size, ga, tint=col)

    def composite(self, rgb, exps, judges, drums=()):
        rgb = np.ascontiguousarray(rgb)
        g = self.geo
        # input-drum press flashes (additive — clean bright pop + glow)
        for is_rim, left, a in drums:
            _add_prescaled(rgb, self._drum_scaled[(is_rim, left)],
                           g.drum_x, g.center_y, a)
        # hit explosions at the target (additive)
        for is_rim, age, big, res in exps:
            if res == "great":
                if age < C.EXPLOSION_GREAT_IN_MS:
                    a = age / C.EXPLOSION_GREAT_IN_MS
                else:
                    f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_GREAT_OUT_MS
                    a = max(0.0, f) ** 4
            else:
                f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_OK_OUT_MS
                a = C.EXPLOSION_OK_PEAK * max(0.0, f)
            if a <= 0.001:
                continue
            _add_prescaled(rgb, self._exp_scaled[(is_rim, big)],
                           g.target_x, g.center_y, a)
        # judgement popups: float up (-0.6→-1.0 pf_h), scale 1→1.4, fade out
        for res, age, rt in judges:
            self._ring_burst(rgb, res, age, rt, g)
            tex = self._judge_tex(res)
            p = age / C.JUDGE_MOVE_MS
            ease = 1.0 - (1.0 - p) ** 5                # OutQuint (move/scale)
            scale = 1.0 + 0.4 * ease
            alpha = max(0.0, (1.0 - p) ** 5)           # FadeOutFromOne, OutQuint
            if alpha <= 0.01:
                continue
            yoff = (0.6 + 0.4 * ease) * g.pf_h
            h0, w0 = tex.shape[0], tex.shape[1]
            # straight alpha for both skin images and Argon text (lazer draws the
            # judgement as a normal SpriteText / Sprite — not additive).
            _blit_straight(rgb, tex, g.target_x, g.center_y - yoff,
                           w0 * scale, h0 * scale, alpha)
        return rgb
