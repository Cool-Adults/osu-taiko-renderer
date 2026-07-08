"""Parse a .osu file into the osu!taiko object stream.

Circle  -> DON (red centre) or KAT (blue rim), by hit-sound; finish -> big note
Slider  -> DRUMROLL spanning [time, time+duration]; finish -> big drumroll
Spinner -> SWELL (denden) over [time, end], with a required hit count

Drumroll duration uses the active timing point's beat length and slider-velocity
multiplier (same as the game). Each note also carries an effective scroll
multiplier (BPM x SV) so notes under different timing scroll at the right speed.
"""
from __future__ import annotations

from pathlib import Path

from .models import TaikoBeatmap, TaikoObject, TaikoType

# HitObject type bitfield
_TYPE_CIRCLE = 1 << 0
_TYPE_SLIDER = 1 << 1
_TYPE_NEW_COMBO = 1 << 2
_TYPE_SPINNER = 1 << 3

# HitSound bitfield
_HS_WHISTLE = 1 << 1
_HS_FINISH = 1 << 2
_HS_CLAP = 1 << 3


class BeatmapParseError(RuntimeError):
    pass


def parse_beatmap(path: Path, *, mods: int = 0, lazer: bool = False) -> TaikoBeatmap:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _split_sections(text)

    diff = _kv(sections.get("Difficulty", ""))
    meta = _kv(sections.get("Metadata", ""))
    general = _kv(sections.get("General", ""))

    cs = _f(diff.get("CircleSize"), 5.0)
    ar = _f(diff.get("ApproachRate"), _f(diff.get("OverallDifficulty"), 5.0))
    od = _f(diff.get("OverallDifficulty"), 5.0)
    hp = _f(diff.get("HPDrainRate"), 5.0)
    slider_mult = _f(diff.get("SliderMultiplier"), 1.4)

    ez = bool(mods & (1 << 1))
    hr = bool(mods & (1 << 4))
    if ez:
        od *= 0.5; hp *= 0.5
    if hr:
        od = min(10.0, od * 1.4); hp = min(10.0, hp * 1.4)
    # Object times stay on the map-time axis (replay frames share it); rate only
    # compresses the output video + atempos audio in render.py.
    dt = bool(mods & (1 << 6)) or bool(mods & (1 << 9))
    ht = bool(mods & (1 << 8))
    rate = 1.5 if dt else (0.75 if ht else 1.0)

    timing = _parse_timing(sections.get("TimingPoints", ""))
    timing.slider_mult = slider_mult        # lazer Velocity = SliderMultiplier
    objects = _parse_hit_objects(
        sections.get("HitObjects", ""),
        timing=timing, slider_mult=slider_mult, od=od,
    )
    objects.sort(key=lambda o: o.time_ms)
    first_t = objects[0].time_ms if objects else 0
    last_t = max((o.end_ms or o.time_ms for o in objects), default=0)
    bar_lines = _generate_bar_lines(timing, first_t, last_t)
    kiai = _parse_kiai(sections.get("TimingPoints", ""), last_t)

    return TaikoBeatmap(
        objects=objects,
        bar_lines=bar_lines,
        kiai_ranges=kiai,
        timing=timing,
        cs=cs, ar=ar, od=od, hp=hp, rate=rate,
        audio_filename=general.get("AudioFilename"),
        background=_parse_background(sections.get("Events", "")),
        breaks=_parse_breaks(sections.get("Events", "")),
        title=meta.get("Title", ""),
        artist=meta.get("Artist", ""),
        version=meta.get("Version", ""),
    )


# --- timing -------------------------------------------------------------------

class _Timing:
    """Resolves beat length and SV multiplier at any time."""

    def __init__(self, points: list[tuple[float, float, bool]]):
        self.points = points
        # base (first uninherited) beat length — the reference BPM for scroll.
        self.base_beat = next((v for _, v, u in points if u and v > 0), 500.0)
        self.slider_mult = 1.0      # set to the map's SliderMultiplier (Velocity)

    def beat_length(self, t: float) -> float:
        bl = self.base_beat
        for time, val, uninh in self.points:
            if time > t:
                break
            if uninh and val > 0:
                bl = val
        return bl

    def sv_mult(self, t: float) -> float:
        mult = 1.0
        for time, val, uninh in self.points:
            if time > t:
                break
            if not uninh and val < 0:
                mult = 100.0 / -val
            elif uninh:
                mult = 1.0
        return mult

    def scroll_mult(self, t: float) -> float:
        """lazer MultiplierControlPoint.Multiplier for taiko (BaseBeatLength is
        the fixed 60-BPM DEFAULT_BEAT_LENGTH=1000, NOT the map's base BPM):
          Multiplier = SliderMultiplier(Velocity) x SV(ScrollSpeed) x 1000/BeatLength.
        Visible time = TimeRange / Multiplier."""
        bl = self.beat_length(t)
        bpm_factor = 1000.0 / bl if bl > 0 else 1.0
        return self.slider_mult * self.sv_mult(t) * bpm_factor


def _parse_timing(block: str) -> _Timing:
    pts: list[tuple[float, float, bool]] = []
    uninh: list[tuple[float, float, int]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        time = _f(parts[0], 0.0)
        beat = _f(parts[1], 500.0)
        uninherited = True if len(parts) < 7 else parts[6].strip() == "1"
        meter = int(_f(parts[2], 4.0)) if len(parts) > 2 else 4
        pts.append((time, beat, uninherited))
        if uninherited and beat > 0:
            uninh.append((time, beat, meter if meter > 0 else 4))
    pts.sort(key=lambda p: p[0])
    uninh.sort(key=lambda p: p[0])
    tm = _Timing(pts)
    tm.uninherited = uninh
    return tm


def _parse_kiai(block: str, last_t: float):
    """Kiai ranges from timing points (effects bit 0). A point with kiai on
    starts a section; it ends at the next point that turns it off."""
    pts = []
    for line in block.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        time = _f(parts[0], 0.0)
        effects = int(_f(parts[7], 0.0)) if len(parts) > 7 else 0
        pts.append((time, bool(effects & 1)))
    pts.sort(key=lambda p: p[0])
    ranges = []
    start = None
    for time, kiai in pts:
        if kiai and start is None:
            start = time
        elif not kiai and start is not None:
            ranges.append((start, time))
            start = None
    if start is not None:
        ranges.append((start, last_t + 1))
    return ranges


def _generate_bar_lines(timing: "_Timing", first_hit: float, last_hit: float):
    """Measure bar lines (BarLineGenerator): one per measure
    (BeatLength × meter) from each uninherited timing point; Major every
    meter-th measure. Returns [(time, scroll_vel, major)]."""
    import math
    pts = getattr(timing, "uninherited", [])
    if not pts:
        return []
    gen_start = min(0.0, first_hit)
    end_all = last_hit + 1.0
    out = []
    for idx, (ptime, beat, meter) in enumerate(pts):
        bar_len = beat * meter
        if bar_len <= 0:
            continue
        end = pts[idx + 1][0] if idx < len(pts) - 1 else end_all + bar_len
        if ptime > gen_start:
            start = ptime
        else:
            n = math.ceil((gen_start - ptime) / bar_len)
            start = ptime + n * bar_len
        beat_i = 0
        t = start
        while t < end + 1e-3:
            out.append((round(t), timing.scroll_mult(t), beat_i % meter == 0))
            t += bar_len
            beat_i += 1
    return out


# --- hit objects --------------------------------------------------------------

def _parse_hit_objects(block: str, *, timing, slider_mult, od) -> list[TaikoObject]:
    out: list[TaikoObject] = []
    started = False
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(",")
        if len(f) < 5:
            continue
        time = int(float(f[2]))
        typ = int(f[3])
        hs = int(f[4])
        is_new = bool(typ & _TYPE_NEW_COMBO) or not started
        started = True
        big = bool(hs & _HS_FINISH)
        scroll = timing.scroll_mult(time)

        if typ & _TYPE_CIRCLE:
            kind = TaikoType.KAT if (hs & (_HS_WHISTLE | _HS_CLAP)) else TaikoType.DON
            out.append(TaikoObject(time, kind, big=big, scroll_vel=scroll,
                                   new_combo=is_new))
        elif typ & _TYPE_SLIDER:
            dur = _slider_duration(f, time, timing, slider_mult)
            out.append(TaikoObject(time, TaikoType.DRUMROLL, big=big,
                                   end_ms=int(time + dur), scroll_vel=scroll,
                                   new_combo=is_new))
        elif typ & _TYPE_SPINNER:
            end = int(float(f[5])) if len(f) > 5 else time + 1000
            hits = _swell_hits(end - time, od)
            out.append(TaikoObject(time, TaikoType.SWELL, end_ms=end,
                                   required_hits=hits, scroll_vel=scroll,
                                   new_combo=is_new))
    return out


def _slider_duration(f, time, timing, slider_mult) -> float:
    """Drumroll length in ms = pixel_length / velocity, velocity from the
    active beat length + SV (same formula osu! uses for sliders)."""
    if len(f) < 8:
        return 0.0
    spans = int(f[6]) if f[6].isdigit() else 1
    pixel_length = _f(f[7], 0.0)
    beat = timing.beat_length(time)
    sv = timing.sv_mult(time)
    velocity = 100.0 * slider_mult * sv / beat if beat > 0 else 0.0
    if velocity <= 0 or pixel_length <= 0:
        return 0.0
    return spans * pixel_length / velocity


def _swell_hits(duration: float, od: float) -> int:
    """Required alternating hits to clear a swell. Approximates osu!taiko's
    duration- and OD-scaled hit count (refine vs lazer once validated)."""
    rate = 3.0 + od * 0.4          # hits/sec, rises with OD
    return max(1, int(duration / 1000.0 * rate))


# --- shared parsing helpers ---------------------------------------------------

def _split_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("[") and line.rstrip().endswith("]"):
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur = line.strip()[1:-1]
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def _kv(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_breaks(events: str) -> list:
    out = []
    for line in events.splitlines():
        f = line.split(",")
        if len(f) >= 3 and f[0].strip() in ("2", "Break"):
            try:
                out.append((int(float(f[1])), int(float(f[2]))))
            except ValueError:
                continue
    return out


def _parse_background(events: str) -> str | None:
    for line in events.splitlines():
        f = line.split(",")
        if len(f) >= 3 and f[0].strip() in ("0", "Background"):
            return f[2].strip().strip('"')
    return None


def _f(s, default: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default
