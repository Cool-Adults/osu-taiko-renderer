"""CLI entrypoint.

    python -m osu_taiko_renderer REPLAY.osr BEATMAP_DIR -o out.mp4 \
        [--resolution 1920x1080] [--fps 60] [--encoder auto] [--skin DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import RenderConfig
from .render import render_taiko


def _resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="osu_taiko_renderer")
    ap.add_argument("osr", type=Path, help="replay .osr file")
    ap.add_argument("beatmap_dir", type=Path, help="dir with .osu + audio + bg")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--resolution", type=_resolution, default=(1920, 1080))
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--encoder", default="auto",
                    help="auto | h264_vaapi | h264_nvenc | libx264")
    ap.add_argument("--encoder-device", default=None, help="e.g. /dev/dri/renderD128")
    ap.add_argument("--skin", type=Path, default=None, help="extracted skin dir")
    ap.add_argument("--default-skin", type=Path, default=None, help="fallback skin dir")
    BA = argparse.BooleanOptionalAction
    ap.add_argument("--skip-intro", action=BA, default=True, help="start at first object")
    ap.add_argument("--results", action=BA, default=True, help="results-screen outro")
    ap.add_argument("--letterbox-breaks", action=BA, default=True)
    ap.add_argument("--pp-counter", action=BA, default=True)
    ap.add_argument("--hit-counter", action=BA, default=True)
    ap.add_argument("--scroll-time", type=int, default=1600,
                    help="ms a 1.0x-SV note is visible (lower = faster scroll)")
    ap.add_argument("--watermark", default="")
    ap.add_argument("--music-volume", type=int, default=100)
    ap.add_argument("--general-volume", type=int, default=100)
    ap.add_argument("--audio-offset", type=int, default=0, help="ms; -earlier")
    ap.add_argument("--bg-dim-intro", type=int, default=0)
    ap.add_argument("--bg-dim-game", type=int, default=70)
    ap.add_argument("--bg-dim-breaks", type=int, default=0)
    ap.add_argument("--bg-blur", type=int, default=0)
    ap.add_argument("--results-seconds", type=float, default=None)
    ap.add_argument("--show-combo", action=BA, default=True)
    ap.add_argument("--show-score", action=BA, default=True)
    ap.add_argument("--show-hp", action=BA, default=True)
    ap.add_argument("--show-grade", action=BA, default=True)
    ap.add_argument("--show-mods", action=BA, default=True)
    ap.add_argument("--logo", action=BA, default=False,
                    help="show_logo: the R3D 'R' tile splash during the intro, "
                         "fading out as gameplay starts (parity with std/catch)")
    # Accept-and-ignore: lets the shared render pipeline pass mode-agnostic
    # flags it also sends to the catch renderer without erroring here.
    args, _unknown = ap.parse_known_args(argv)

    cfg = RenderConfig(
        resolution=args.resolution,
        fps=args.fps,
        encoder=args.encoder,
        encoder_device=args.encoder_device,
        skin_dir=args.skin,
        default_skin_dir=args.default_skin,
        skip_intro=args.skip_intro,
        show_results=args.results,
        letterbox_breaks=args.letterbox_breaks,
        scroll_time_ms=args.scroll_time,
        show_pp_counter=args.pp_counter,
        show_hit_counter=args.hit_counter,
        watermark=args.watermark,
        music_volume=args.music_volume,
        general_volume=args.general_volume,
        audio_offset_ms=args.audio_offset,
        bg_dim_intro=args.bg_dim_intro,
        bg_dim_game=args.bg_dim_game,
        bg_dim_breaks=args.bg_dim_breaks,
        bg_blur=args.bg_blur,
        show_combo=args.show_combo,
        show_score=args.show_score,
        show_hp_bar=args.show_hp,
        show_grade=args.show_grade,
        show_mods=args.show_mods,
        show_logo=args.logo,
    )
    if args.results_seconds is not None:
        cfg.results_ms = int(args.results_seconds * 1000)

    def progress(pct: int) -> None:
        print(f"\rrendering… {pct:3d}%", end="", file=sys.stderr, flush=True)

    out = render_taiko(args.osr, args.beatmap_dir, args.output, cfg,
                       progress_callback=progress)
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
