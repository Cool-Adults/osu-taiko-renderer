"""Phase 1 orchestrator: parse -> simulate -> per-frame GL draw + HUD -> ffmpeg.

Owns a small ffmpeg subprocess (raw rgb24 on stdin) so it stays decoupled
from osu_renderer's encode FIFO machinery. HUD text is composited on the CPU
with PIL after GL readback — cheap and avoids a GL text pass for Phase 1.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .assets import build_textures
from .beatmap import parse_beatmap
from .gl import SpriteRenderer
from .models import RenderConfig
from .replay import parse_replay
from .scene import TaikoSim


class TaikoRenderError(RuntimeError):
    pass


def render_taiko(
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    cfg: RenderConfig | None = None,
    *,
    progress_callback=None,
) -> Path:
    cfg = cfg or RenderConfig()
    frames, meta = parse_replay(osr_path)
    osu_path = _find_osu(beatmap_dir, meta.beatmap_md5)
    bm = parse_beatmap(osu_path, mods=meta.mods)
    if not bm.objects:
        raise TaikoRenderError(f"no hit objects parsed from {osu_path.name}")
    audio = bm.audio_filename and (beatmap_dir / bm.audio_filename)
    audio = audio if (audio and audio.is_file()) else None
    bg = bm.background and (beatmap_dir / bm.background)
    bg = bg if (bg and bg.is_file()) else None
    return render_core(bm, frames, meta, output_path, cfg, audio=audio, bg=bg,
                       progress_callback=progress_callback, osu_path=osu_path)


def render_core(
    bm,
    frames,
    meta,
    output_path: Path,
    cfg: RenderConfig,
    *,
    audio: Path | None = None,
    bg: Path | None = None,
    progress_callback=None,
    osu_path: Path | None = None,
) -> Path:
    """Render from already-parsed beatmap/frames/meta. Shared by the osr path
    and tests."""
    from .fonts import set_skin_font
    set_skin_font(cfg.skin_dir)

    # Phase 1: procedural taiko textures + simple HUD (no skin asset wiring yet).
    skin = None
    sim = TaikoSim(bm, frames, cfg, skin=skin, has_bg=bg is not None, meta=meta)
    if cfg.show_pp_counter and osu_path is not None:
        sim.compute_pp_curve(osu_path, meta.mods)
    # preempt = the first object's actual on-screen travel time (scroll_time is
    # the SV=1 base; real notes scroll faster, so the visible time is
    # scroll_time / scroll_vel). Using the raw scroll_time left ~5s of empty
    # playfield before the first note entered.
    first = bm.objects[0].time_ms
    # End of the map = the latest object END (a trailing swell/drumroll runs
    # well past its start), not just the last object's start — otherwise the
    # gameplay tail (and the swell's clear animation) gets clipped. Matches
    # the same computation TaikoSim uses for its life-bar/fail gating.
    last = max((o.end_ms or o.time_ms) for o in bm.objects)
    first_sv = max(getattr(bm.objects[0], "scroll_vel", 1.0) or 1.0, 0.1)
    preempt = sim.geo.scroll_time / first_sv
    # skip_intro: start at the first object's approach; else render the full
    # intro from the song start.
    lead = min(cfg.lead_in_ms, 800)        # brief lead-in, not a long empty gap
    # Cap the first note's visible approach when skipping the intro: a low-SV /
    # low-BPM first note has a huge travel time (preempt), which otherwise made
    # the render open on several seconds of an almost-empty playfield while one
    # note slowly crawled in. Start closer so the intro is always tight (~2s).
    approach = min(preempt, 2000.0)
    if cfg.skip_intro:
        start_ms = int(first - approach - lead)
    else:
        start_ms = min(0, int(first - preempt - lead))
    # intro R3D splash window opens at the render's first frame (no seizure
    # card in taiko, so it begins immediately -- std offsets by the seizure
    # duration). The sim fades it out at the first note's scroll-in
    # (sim.first_spawn_ms), the same "first approach" the std/catch use.
    sim.logo_start_ms = start_ms if cfg.show_logo else None
    # A failed/quit replay stops recording at death; end the video there rather
    # than playing the rest of the map (sim flags this from the life-bar / where
    # the replay frames stop).
    if getattr(sim, "failed", False) and getattr(sim, "fail_time_ms", 0):
        gameplay_end_ms = int(sim.fail_time_ms + 700)
    else:
        gameplay_end_ms = int(last + cfg.tail_ms)
    # results outro (matches osu_renderer: 800ms gap, then the card) — on by default
    RESULTS_GAP_MS, FADE_MS = 800, 400
    if cfg.show_results:
        results_start_ms = gameplay_end_ms + RESULTS_GAP_MS
        total_end_ms = results_start_ms + cfg.results_ms
    else:
        results_start_ms = total_end_ms = gameplay_end_ms
    # DT/HT playback: the simulation lives on the map-time axis, but a DT play
    # should *look* 1.5x faster. So gameplay frames advance map-time by
    # frame_ms*rate per output frame (fewer frames at the same fps => sped up),
    # and the audio is atempo'd by the same rate. The results outro stays
    # real-time for cross-mode consistency with the mania renderer.
    rate = getattr(bm, "rate", 1.0) or 1.0
    frame_ms = 1000.0 / cfg.fps
    map_step = frame_ms * rate
    gameplay_frames = max(1, int((gameplay_end_ms - start_ms) / map_step))
    outro_frames = max(0, int((total_end_ms - gameplay_end_ms) / frame_ms)) if cfg.show_results else 0
    n_frames = gameplay_frames + outro_frames

    w, h = cfg.resolution
    renderer = SpriteRenderer(w, h)
    if skin is not None:
        for key, rgba in skin.textures.items():
            renderer.upload_texture(key, rgba)
    else:
        for key, rgba in build_textures(cfg.skin_dir).items():
            renderer.upload_texture(key, rgba)
    from .assets import bake_logo_tile, logo_glow_rgba
    renderer.upload_texture("logo_tile", bake_logo_tile())
    renderer.upload_texture("logo_glow", logo_glow_rgba())
    if bg is not None:
        renderer.upload_texture("bg", _bg_cover(bg, w, h, cfg.bg_blur))

    total_dur_s = n_frames / cfg.fps
    proc = _spawn_ffmpeg(cfg, output_path, audio, start_ms, rate, total_dur_s)
    # HUD: legacy (true-to-skin) when the skin ships a score font, else Argon.
    from .argon.hud import ArgonHud
    from .skin_hud import LegacyHud
    _lh = LegacyHud(cfg.resolution, meta, bm, first, last, sim, sim.skin)
    if _lh.has_fonts():
        hud = _lh
    else:
        hud = ArgonHud(cfg.resolution, meta, bm, first, last, sim, cfg=cfg)
    from .argon.compositor import ArgonEffects, bloom as _bloom
    effects = ArgonEffects(sim.geo, cfg.skin_dir)

    from .hud import draw_results
    last_gameplay = None
    try:
        for i in range(n_frames):
            if i < gameplay_frames:
                t = int(start_ms + i * map_step)
                scene = sim.build_scene(t)
                renderer.begin()
                renderer.draw(scene.sprites)
                rgb = renderer.read_rgb()
                exps, judges = sim.active_effects(t)
                rgb = effects.composite(rgb, exps, judges, sim.drum_flashes(t))
                rgb = hud.overlay(rgb, scene)
                last_gameplay = rgb
            else:
                # outro: frozen final gameplay frame, then the results card
                # fades in (consistent with the mania renderer). Real-time.
                t = int(gameplay_end_ms + (i - gameplay_frames) * frame_ms)
                rgb = last_gameplay.copy() if last_gameplay is not None else \
                    renderer.read_rgb()
                if cfg.show_results and t >= results_start_ms:
                    op = min(1.0, (t - results_start_ms) / FADE_MS)
                    rgb = hud.draw_results(rgb, op)
            try:
                proc.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                break
            if progress_callback and i % cfg.fps == 0:
                progress_callback(int(i / n_frames * 100))
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        ret = proc.wait()
        renderer.release()

    if ret != 0:
        tail = ""
        errlog = getattr(proc, "_catch_errlog", None)
        if errlog and Path(errlog).exists():
            tail = Path(errlog).read_text(errors="replace")[-800:]
        raise TaikoRenderError(f"ffmpeg exited {ret}\n{tail}")
    if not output_path.exists() or output_path.stat().st_size < 8_000:
        raise TaikoRenderError("output too small / missing — render likely failed")
    if progress_callback:
        progress_callback(100)
    return output_path


# --- ffmpeg -------------------------------------------------------------------

def _probe_encoder(cfg: RenderConfig) -> tuple[str, str | None]:
    if cfg.encoder != "auto":
        # vaapi always needs a device for the hwupload filter; default it.
        if cfg.encoder == "h264_vaapi":
            return cfg.encoder, cfg.encoder_device or "/dev/dri/renderD128"
        return cfg.encoder, cfg.encoder_device
    dev = cfg.encoder_device or "/dev/dri/renderD128"
    if Path(dev).exists() and _ffmpeg_has("h264_vaapi"):
        return "h264_vaapi", dev
    if _ffmpeg_has("h264_nvenc"):
        return "h264_nvenc", None
    return "libx264", None


def _ffmpeg_has(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return False
    return name in out


def _spawn_ffmpeg(cfg: RenderConfig, output_path: Path, audio: Path | None,
                  start_ms: int, rate: float = 1.0, total_dur_s: float | None = None):
    w, h = cfg.resolution
    enc, dev = _probe_encoder(cfg)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if enc == "h264_vaapi" and dev:
        cmd += ["-vaapi_device", dev]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(cfg.fps),
            "-i", "pipe:0"]
    if audio is not None:
        cmd += ["-i", str(audio)]

    # video codec + pixel path
    if enc == "h264_vaapi":
        cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-b:v", "8M"]
    elif enc == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p", "-b:v", "8M"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "20"]

    if audio is not None:
        af = _audio_filter(start_ms, rate, total_dur_s,
                           music_volume=cfg.music_volume,
                           general_volume=cfg.general_volume,
                           audio_offset_ms=cfg.audio_offset_ms)
        if af:
            cmd += ["-af", af]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

    # web-streamable: move the moov atom to the front so browsers/iOS can
    # play before the whole file downloads (loudnorm re-adds this, but be
    # robust if that post-step is skipped/fails).
    cmd += ["-movflags", "+faststart", str(output_path)]
    import tempfile
    errf = tempfile.NamedTemporaryFile(
        prefix="catch_ffmpeg_", suffix=".log", delete=False, mode="w+",
    )
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf, bufsize=0)
    proc._catch_errlog = errf.name  # type: ignore[attr-defined]
    return proc


def _audio_filter(start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  music_volume: int = 100, general_volume: int = 100,
                  audio_offset_ms: int = 0) -> str:
    """Speed the song to the mod rate (DT/HT), then align so video t=0 is
    `start_ms` into the rate-adjusted song. Applies preset volume + offset."""
    parts = []
    if abs(rate - 1.0) > 1e-3:
        parts.append(f"atempo={rate:.4f}")  # speed only (no pitch shift)
    # start_ms is in MAP time; after atempo the song plays at map/rate, so the
    # real offset where video t=0 lands is start_ms/rate. audio_offset shifts the
    # song vs gameplay (negative = audio earlier).
    real_start = (start_ms - audio_offset_ms) / rate
    if real_start > 0:
        parts.append(f"atrim=start={real_start / 1000:.3f}")
        parts.append("asetpts=PTS-STARTPTS")
    elif real_start < 0:
        parts.append(f"adelay={int(-real_start)}:all=1")
    # Pad with silence so the audio spans the full video (incl. the results
    # outro past the song's end). Bound the pad to the exact video duration —
    # an UNBOUNDED apad races the (slow) raw-video pipe and overflows the
    # filtergraph buffer (ffmpeg reports it as ENOSPC and dies).
    # Loudness-normalise to a consistent EBU R128 baseline (single-pass) so
    # hot beatmap masters stop blasting: I=-14 LUFS, true-peak -1.5 dBTP.
    # The volume trim below is applied AFTER, relative to this baseline.
    parts.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    vol = (general_volume / 100.0) * (music_volume / 100.0)
    if abs(vol - 1.0) > 1e-3:
        parts.append(f"volume={max(0.0, vol):.3f}")
    if total_dur_s and total_dur_s > 0:
        parts.append(f"apad=whole_dur={total_dur_s:.3f}")
    else:
        parts.append("apad")
    return ",".join(parts)


def _bg_cover(path: Path, w: int, h: int, blur: int = 0) -> "np.ndarray":
    """Load the beatmap background and cover-crop it to WxH (no distortion)."""
    im = Image.open(path).convert("RGB")
    scale = max(w / im.width, h / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    im = im.crop((left, top, left + w, top + h))
    if blur and blur > 0:
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    return np.array(im)


def _find_osu(beatmap_dir: Path, md5: str) -> Path:
    osus = sorted(beatmap_dir.glob("*.osu"))
    if not osus:
        raise TaikoRenderError(f"no .osu in {beatmap_dir}")
    if md5:
        for p in osus:
            if hashlib.md5(p.read_bytes()).hexdigest() == md5:
                return p
    # fall back to a Mode:1 (taiko) beatmap, else the first
    for p in osus:
        head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        if "Mode: 1" in head or "Mode:1" in head:
            return p
    return osus[0]


# --- HUD ----------------------------------------------------------------------

class _Hud:
    def __init__(self, w, h, meta, bm):
        self.w, self.h = w, h
        self.meta = meta
        self.bm = bm
        big = max(20, int(h * 0.07))
        med = max(16, int(h * 0.035))
        small = max(12, int(h * 0.025))
        self.f_combo = _font(big)
        self.f_score = _font(med)
        self.f_small = _font(small)

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        img = Image.fromarray(rgb, "RGB")
        d = ImageDraw.Draw(img)
        # combo bottom-left
        if scene.combo > 0:
            d.text((int(self.w * 0.02), int(self.h * 0.86)), f"{scene.combo}x",
                   font=self.f_combo, fill=(255, 255, 255))
        # score top-right
        d.text((int(self.w * 0.98), int(self.h * 0.03)), f"{scene.score:,}",
               font=self.f_score, fill=(255, 255, 255), anchor="ra")
        # player + title top-left
        d.text((int(self.w * 0.02), int(self.h * 0.03)), self.meta.player_name,
               font=self.f_small, fill=(230, 230, 240))
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        d.text((int(self.w * 0.02), int(self.h * 0.065)), title,
               font=self.f_small, fill=(180, 180, 200))
        # hp bar top center
        bx, by, bw, bh = int(self.w * 0.30), int(self.h * 0.02), int(self.w * 0.40), 10
        d.rectangle([bx, by, bx + bw, by + bh], fill=(40, 40, 50))
        d.rectangle([bx, by, bx + int(bw * scene.hp), by + bh], fill=(120, 220, 140))
        return np.asarray(img)


from .fonts import font as _font  # skin-aware, host-robust font resolver
