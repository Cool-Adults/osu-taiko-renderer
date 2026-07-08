"""Decode an osu!taiko .osr into per-frame don/kat key state.

Taiko replays store the standard .osr key bitfield. Empirically verified
against lazer taiko replays (player 'nichijou', a clean 154-note FC):
  bit 1 (M1) = DON (centre), bit 2 (M2) = KAT (rim) — every don press hit a
  don note, every kat press a kat note.
The second key pair (K1=4, K2=8) is treated symmetrically (K1 don, K2 kat) —
the common osu!taiko config; pending a multi-config replay to nail it exactly,
this is correct for single-pair players and a sane default otherwise. A big
note (finish) is hit by pressing both centre or both rim keys at once.
"""
from __future__ import annotations

from pathlib import Path

from osrparse import Replay

from .models import ReplayMeta, TaikoFrame

_SEED_DELTA = -12345
_DON_BITS = 1 | 4   # M1, K1 (centre)
_KAT_BITS = 2 | 8   # M2, K2 (rim)


class ReplayParseError(RuntimeError):
    pass


def parse_replay(path: Path) -> tuple[list[TaikoFrame], ReplayMeta]:
    if not path.exists():
        raise ReplayParseError(f"replay not found: {path}")
    try:
        r = Replay.from_path(path)
    except Exception as e:  # noqa: BLE001 - osrparse raises bare exceptions
        raise ReplayParseError(f"osrparse failed: {e}") from e

    frames: list[TaikoFrame] = []
    t = 0
    first = True
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        if delta == _SEED_DELTA:
            continue
        if first:
            first = False
            if delta < -5000:   # garbage placeholder frame
                delta = 0
        t += delta
        k = getattr(ev, "keys", 0)
        k = int(getattr(k, "value", k))
        cl, cr = bool(k & 1), bool(k & 4)   # centre-left / centre-right (don)
        rl, rr = bool(k & 2), bool(k & 8)   # rim-left / rim-right (kat)
        don = cl or cr
        kat = rl or rr
        big = (cl and cr) or (rl and rr)
        frames.append(TaikoFrame(time_ms=max(t, 0), don=don, kat=kat, big=big,
                                 cl=cl, cr=cr, rl=rl, rr=rr))
    frames.sort(key=lambda f: f.time_ms)
    replay_end_ms = frames[-1].time_ms if frames else 0

    # Life-bar graph: osu! records the player's actual HP over time. When present
    # it's the ground-truth HP bar (exactly true-to-game). osrparse exposes it as
    # a list of LifeBarState(time, life); it's often empty for lazer/API replays.
    lb = getattr(r, "life_bar_graph", None) or []
    life_bar = tuple((int(getattr(s, "time", 0)), float(getattr(s, "life", 0.0)))
                     for s in lb)

    total = r.count_300 + r.count_100 + r.count_miss
    if total > 0:
        acc = (r.count_300 + r.count_100 * 0.5) / total
    else:
        acc = 1.0
    meta = ReplayMeta(
        mode=int(r.mode.value if hasattr(r.mode, "value") else r.mode),
        beatmap_md5=str(getattr(r, "beatmap_hash", "") or ""),
        player_name=r.username,
        mods=int(r.mods),
        score=int(r.score),
        max_combo=int(r.max_combo),
        count_300=int(r.count_300),
        count_100=int(r.count_100),
        count_50=int(r.count_50),
        count_katu=int(r.count_katu),
        count_miss=int(r.count_miss),
        accuracy=round(acc * 100, 2),
        grade=_grade(acc, r),
        game_version=int(getattr(r, "game_version", 0) or 0),
        life_bar=life_bar,
        replay_end_ms=replay_end_ms,
    )
    return frames, meta


def hit_events(frames: list[TaikoFrame]) -> list[tuple[int, str, bool]]:
    """Rising-edge hits: (time_ms, 'don'|'kat', is_big). A don and kat rising
    on the same frame both count (a big hit / simultaneous)."""
    out: list[tuple[int, str, bool]] = []
    pd = pk = False
    for f in frames:
        if f.don and not pd:
            out.append((f.time_ms, "don", f.big))
        if f.kat and not pk:
            out.append((f.time_ms, "kat", f.big))
        pd, pk = f.don, f.kat
    return out


def _grade(acc: float, r) -> str:
    """osu!lazer rank (ScoreProcessor.RankFromScore): accuracy-only thresholds
    SS=100%, S>=95%, A>=90%, B>=80%, C>=70%, else D. (Silver SS/S for HD/FL is
    applied at draw time by colour.)"""
    if acc >= 1.0:
        return "SS"
    if acc >= 0.95:
        return "S"
    if acc >= 0.90:
        return "A"
    if acc >= 0.80:
        return "B"
    if acc >= 0.70:
        return "C"
    return "D"
