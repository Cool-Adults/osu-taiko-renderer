"""Regression tests for the osu!stable replay lead-in fix (see
osu_taiko_renderer/replay._recover_leadin_offset).

osrparse silently drops the up-to-two leading (256,-500) placeholder frames
WITHOUT accumulating their deltas, discarding the audio lead-in / intro-skip
that osu!'s LegacyScoreDecoder folds into the running clock. Accumulating the
survivors from 0 then starts the key timeline the whole lead-in too early ->
every don/kat judged against the wrong note (mass miss).

The committed real fixture (stable_leadin.osr, 16863ms intro-skip) happens to be
the "cancel" variety osu! returns to ~0 with a <-5000ms first frame, so the OLD
-5000 guard only mistimed it by ~15ms; the CATASTROPHIC variety (a lead-in NOT
cancelled, which the old guard shifted by the whole intro-skip) is exercised by
test_recover_leadin_synthetic below. Both are now exact.

Runnable two ways:  pytest tests/test_leadin.py   OR   python tests/test_leadin.py
"""
from __future__ import annotations

import lzma
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.replay import _recover_leadin_offset, parse_replay

HERE = Path(__file__).resolve().parent
STABLE_LEADIN = HERE / "stable_leadin.osr"     # real corpus replay, 16863ms intro-skip
_STABLE_LEADIN_MS = 16863


def _uleb_string(s: str) -> bytes:
    b = s.encode("utf-8")
    out = bytearray([0x0b])
    n = len(b)
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            break
    return bytes(out) + b


def _make_osr(frames: str, mode: int = 1) -> bytes:
    blob = lzma.compress(frames.encode("ascii"), format=lzma.FORMAT_ALONE)
    out = bytearray()
    out.append(mode)
    out += struct.pack("<i", 20260101)
    out += _uleb_string("beatmapmd5")
    out += _uleb_string("player")
    out += _uleb_string("replaymd5")
    out += struct.pack("<6h", 0, 0, 0, 0, 0, 0)
    out += struct.pack("<i", 0)
    out += struct.pack("<h", 0)
    out += struct.pack("<b", 0)
    out += struct.pack("<i", 0)
    out += _uleb_string("")
    out += struct.pack("<q", 0)
    out += struct.pack("<i", len(blob))
    out += blob
    out += struct.pack("<q", 0)
    return bytes(out)


def test_recover_leadin_synthetic(tmp_path):
    # a 2342ms lead-in NOT cancelled by the first real frame -> the catastrophic
    # case the old -5000 guard shifted by the whole intro-skip.
    p = tmp_path / "s.osr"
    p.write_bytes(_make_osr("0|256|-500|0,2342|256|-500|0,14|1|1|1,20|1|1|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 2342


def test_recover_leadin_lazer_synthetic(tmp_path):
    p = tmp_path / "l.osr"
    p.write_bytes(_make_osr("0|1|1|1,16|1|1|0,17|1|1|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 0


def test_recover_leadin_single_placeholder(tmp_path):
    p = tmp_path / "one.osr"
    p.write_bytes(_make_osr("0|256|-500|0,50|1|1|1,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 0


def test_recover_leadin_failsoft(tmp_path):
    p = tmp_path / "junk.osr"
    p.write_bytes(b"not an osr file at all")
    assert _recover_leadin_offset(p) == 0


def test_real_stable_leadin_recovered():
    assert _recover_leadin_offset(STABLE_LEADIN) == _STABLE_LEADIN_MS


def test_parse_seeds_clock_matches_osu():
    """End-to-end on a real stable replay: the parsed key clock must start at
    the osu!-correct time (seed + first survivor delta), which the OLD from-0
    logic got wrong."""
    from osrparse import Replay
    seed = _recover_leadin_offset(STABLE_LEADIN)
    r = Replay.from_path(STABLE_LEADIN)
    deltas = [int(e.time_delta) for e in (r.replay_data or [])
              if int(e.time_delta) != -12345]
    expected_first = max(0, seed + deltas[0])
    old_first = max(0, 0 if deltas[0] < -5000 else deltas[0])

    frames, _ = parse_replay(STABLE_LEADIN)
    assert frames == sorted(frames, key=lambda f: f.time_ms)
    assert all(f.time_ms >= 0 for f in frames)
    assert min(f.time_ms for f in frames) == expected_first
    assert old_first != expected_first


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    test_recover_leadin_synthetic(tmp)
    test_recover_leadin_lazer_synthetic(tmp)
    test_recover_leadin_single_placeholder(tmp)
    test_recover_leadin_failsoft(tmp)
    test_real_stable_leadin_recovered()
    test_parse_seeds_clock_matches_osu()
    print("all taiko lead-in tests PASSED")
