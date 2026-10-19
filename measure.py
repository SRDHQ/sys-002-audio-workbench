#!/usr/bin/env python3
"""WAV file reader and signal measurement functions for the audio workbench (M04, D-043, D-045).

Stdlib only. Does not import or call `synth.py` or `generate.py` (AC8).
Measures WAV header fields, raw frame counts, rail counts (|s| == 32767),
-32768 sample counts, upward zero crossings with linear interpolation,
mean frequency over analysis windows, peak values, and nonzero sample positions.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import wave

RAIL_LEVEL = 32767
FORBIDDEN_MIN = -32768


@dataclass(frozen=True)
class WavMeasurement:
    """Measured header and sample data from a single WAV file."""

    ok: bool
    error: str | None = None
    nchannels: int = 0
    sampwidth: int = 0
    framerate: int = 0
    comptype: str = ""
    nframes_header: int = 0
    raw_bytes_len: int = 0
    samples: array | None = None
    rail_count: int = 0
    neg32768_count: int = 0


def upward_crossings(
    samples: Sequence[int],
    w_start: int = 0,
    w_end: int | None = None,
) -> list[float]:
    """Return linearly interpolated positions of upward zero crossings in [w_start, w_end].

    An upward crossing is an index `n` with `s[n] < 0 <= s[n+1]`, where both `n` and `n+1`
    lie inside the inclusive analysis window `[w_start, w_end]`.
    Its position in samples is `x = n + s[n] / (s[n] - s[n+1])`.
    When `s[n+1] == 0`, the crossing is at `n + 1.0`.
    A 0-valued sample on a falling slope (`s[n] >= 0 > s[n+1]` or `s[n] == 0 > s[n+1]`)
    is not an upward crossing.
    """
    if not samples:
        return []
    last_idx = len(samples) - 1 if w_end is None else min(w_end, len(samples) - 1)
    first_idx = max(0, w_start)
    crossings: list[float] = []
    for n in range(first_idx, last_idx):
        s0 = samples[n]
        s1 = samples[n + 1]
        if s0 < 0 and s1 >= 0:
            x = n + s0 / (s0 - s1)
            crossings.append(float(x))
    return crossings


def estimate_frequency(
    samples: Sequence[int],
    sample_rate: int = 48000,
    w_start: int = 0,
    w_end: int | None = None,
) -> tuple[int, float]:
    """Estimate mean frequency over `[w_start, w_end]` from upward crossings.

    Returns `(K, f_hat)` where `K` is the number of upward crossings and
    `f_hat = sample_rate * (K - 1) / (x_K - x_1)`.
    Raises `ValueError` if fewer than 2 upward crossings are found (never returns 0 Hz).
    """
    xs = upward_crossings(samples, w_start=w_start, w_end=w_end)
    k = len(xs)
    if k < 2:
        raise ValueError(f"fewer than 2 upward crossings ({k} found)")
    span = xs[-1] - xs[0]
    if span <= 0.0:
        raise ValueError("non-positive crossing span")
    f_hat = sample_rate * (k - 1) / span
    return (k, f_hat)


def measure_peaks(samples: Sequence[int]) -> tuple[int, int]:
    """Return `(max_sample, min_sample)` for non-empty `samples`."""
    if not samples:
        raise ValueError("empty sample sequence")
    return (max(samples), min(samples))


def measure_nonzero(samples: Sequence[int]) -> list[tuple[int, int]]:
    """Return `(index, value)` for every nonzero sample in `samples`."""
    return [(idx, int(val)) for idx, val in enumerate(samples) if val != 0]


def measure_largest_non_rail(samples: Sequence[int]) -> int:
    """Return the largest `|s|` strictly below `RAIL_LEVEL` (32767), or 0 if none."""
    best = 0
    for s in samples:
        a = abs(int(s))
        if a < RAIL_LEVEL and a > best:
            best = a
    return best


def read_and_measure_wav(path: str | Path) -> WavMeasurement:
    """Open a WAV file at `path`, read its header and frames, and compute base measurements.

    Never raises on missing/corrupt/wrong-format files; returns `WavMeasurement(ok=False, error=...)`.
    Note: `wave` in Python 3.14 handles byte-swapping on big-endian machines inside `readframes`,
    so `array('h')` from `raw_bytes` is already in native byte order and must not be swapped again.
    """
    p = Path(path)
    try:
        with p.open("rb") as fobj:
            with wave.open(fobj, "rb") as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                comptype = wf.getcomptype()
                nframes_header = wf.getnframes()
                raw_bytes = wf.readframes(nframes_header + 1)
    except FileNotFoundError:
        return WavMeasurement(ok=False, error="missing")
    except (wave.Error, EOFError, OSError) as exc:
        reason = type(exc).__name__
        return WavMeasurement(ok=False, error=f"read_error:{reason}")

    raw_len = len(raw_bytes)
    fmt_ok = (
        nchannels == 1
        and sampwidth == 2
        and framerate == 48000
        and comptype == "NONE"
    )
    if not fmt_ok:
        return WavMeasurement(
            ok=False,
            error=f"bad_format:ch={nchannels},sw={sampwidth},rate={framerate},comp={comptype}",
            nchannels=nchannels,
            sampwidth=sampwidth,
            framerate=framerate,
            comptype=comptype,
            nframes_header=nframes_header,
            raw_bytes_len=raw_len,
        )

    if raw_len % 2 != 0:
        return WavMeasurement(
            ok=False,
            error=f"odd_byte_count:{raw_len}",
            nchannels=nchannels,
            sampwidth=sampwidth,
            framerate=framerate,
            comptype=comptype,
            nframes_header=nframes_header,
            raw_bytes_len=raw_len,
        )

    samples = array("h")
    samples.frombytes(raw_bytes)
    rail_count = sum(1 for s in samples if abs(s) == RAIL_LEVEL)
    neg32768_count = sum(1 for s in samples if s == FORBIDDEN_MIN)

    return WavMeasurement(
        ok=True,
        error=None,
        nchannels=nchannels,
        sampwidth=sampwidth,
        framerate=framerate,
        comptype=comptype,
        nframes_header=nframes_header,
        raw_bytes_len=raw_len,
        samples=samples,
        rail_count=rail_count,
        neg32768_count=neg32768_count,
    )
