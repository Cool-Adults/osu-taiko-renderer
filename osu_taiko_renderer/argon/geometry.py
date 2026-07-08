"""Playfield geometry derived from osu!lazer's TaikoPlayfield layout.

All positions/sizes trace to source constants in `_const`:
- note Ø = DEFAULT_SIZE × playfield_height
- playfield height = relative-height × screen_height (16:9 → BASE_HEIGHT/768)
- drum centre x = (INPUT_DRUM_WIDTH/2) × local-scale
- hit-target x = (INPUT_DRUM_WIDTH + hit_target_width/2 + hit_target_offset) × local-scale
"""
from __future__ import annotations

from dataclasses import dataclass

from . import _const as C

_MAX_ASPECT = 16.0 / 9.0
_MIN_ASPECT = 5.0 / 4.0


@dataclass(frozen=True)
class Geometry:
    w: int
    h: int
    scale: float          # local (768-ref) → px
    pf_h: float           # playfield height px
    center_y: float       # hit row y px
    drum_x: float         # input-drum centre x px
    drum_d: float         # input-drum visible diameter px
    target_x: float       # hit-target / note-judge x px
    note_d: float         # normal note diameter px
    big_d: float          # big (strong) note diameter px
    scroll_time: float    # ms a SV=1 (base BPM) note is visible (lazer TimeRange)

    @property
    def hit_target_d(self) -> float:
        return self.note_d


def compute(w: int, h: int) -> Geometry:
    aspect = w / h
    base_rel = C.BASE_HEIGHT / C.REF_SCREEN_H            # 0.2604…
    rel = base_rel
    if aspect > _MAX_ASPECT:
        rel *= aspect / _MAX_ASPECT
    elif aspect < _MIN_ASPECT:
        rel *= aspect / _MIN_ASPECT
    rel = min(rel, 1.0 / 3.0)
    pf_h = rel * h
    scale = (h / C.REF_SCREEN_H) * (rel / base_rel)     # = h/768 at 16:9
    center_y = (C.PLAYFIELD_TOP_FRAC + rel / 2.0) * h
    drum_x = (C.INPUT_DRUM_WIDTH / 2.0) * scale
    drum_d = C.INPUT_DRUM_WIDTH * scale * C.DRUM_INNER_SCALE
    hit_target_width = C.BASE_HEIGHT
    target_x = (C.INPUT_DRUM_WIDTH + hit_target_width / 2.0
                + C.HIT_TARGET_OFFSET) * scale
    note_d = C.DEFAULT_SIZE * pf_h
    big_d = C.DEFAULT_STRONG_SIZE * pf_h
    # lazer TaikoPlayfieldAdjustmentContainer.ComputeTimeRange (Overlapping):
    # how long a SV=1 note (at base BPM) is visible. Aspect-clamped to [5/4,16/9].
    ca = min(max(aspect, _MIN_ASPECT), _MAX_ASPECT)
    in_length = ca * 480.0 - 160.0
    scroll_time = (in_length / 100.0 * 1000.0) / 1.4      # /VELOCITY_MULTIPLIER
    return Geometry(w=w, h=h, scale=scale, pf_h=pf_h, center_y=center_y,
                    drum_x=drum_x, drum_d=drum_d, target_x=target_x,
                    note_d=note_d, big_d=big_d, scroll_time=scroll_time)
