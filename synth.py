#!/usr/bin/env python3
"""Generator synthesis for the five audio workbench test signals (M04, D-043, D-045).

Stdlib only. Computes 16-bit signed integer PCM samples according to the
Architect specification (revision 1):
- Sample rate: 48,000 Hz
- Sample count: N = round(48000 * duration), sample index n in 0 .. N-1 at t_n = n / 48000
- Float-to-int PCM conversion:
  1. Hard clip: y_c = max(-1.0, min(1.0, y))
  2. Scale and round (ties to even): s = round(32767 * y_c)
  3. Saturate (safety only): max(-32767, min(32767, s))
  Symmetric rails +/-32767; -32768 is never written.
"""
from __future__ import annotations

from array import array
import math

SAMPLE_RATE = 48000
FULL_SCALE = 32767

FILE_NAMES: tuple[str, ...] = (
    "sine_1k.wav",
    "sweep_log_20_20k.wav",
    "clicks_10.wav",
    "silence.wav",
    "sine_1k_clipped_2x.wav",
)


def quantize_sample(y: float) -> int:
    """Convert a full-scale float sample y to a 16-bit symmetric PCM integer."""
    y_c = max(-1.0, min(1.0, y))
    s = round(FULL_SCALE * y_c)
    return max(-FULL_SCALE, min(FULL_SCALE, s))


def synth_sine_1k() -> array:
    """1.0 s, 1 kHz sine at amplitude 0.5 with half-sample phase offset."""
    n_samples = round(SAMPLE_RATE * 1.0)
    two_pi_f_over_r = 2.0 * math.pi * 1000.0 / SAMPLE_RATE
    return array(
        "h",
        (quantize_sample(0.5 * math.sin(two_pi_f_over_r * (n + 0.5))) for n in range(n_samples)),
    )


def synth_sweep_log_20_20k() -> array:
    """2.0 s logarithmic sweep from 20 Hz to 20 kHz at amplitude 0.5, starting at phase 0."""
    duration = 2.0
    n_samples = round(SAMPLE_RATE * duration)
    f1 = 20.0
    f2 = 20000.0
    c = math.log(f2 / f1) / duration
    coeff = 2.0 * math.pi * f1 / c
    return array(
        "h",
        (
            quantize_sample(0.5 * math.sin(coeff * (math.exp(c * (n / SAMPLE_RATE)) - 1.0)))
            for n in range(n_samples)
        ),
    )


def synth_clicks_10() -> array:
    """1.0 s click train: all zeros except 10 single-sample impulses of integer +16000."""
    n_samples = round(SAMPLE_RATE * 1.0)
    samples = array("h", [0] * n_samples)
    for k in range(10):
        idx = 2400 + 4800 * k
        samples[idx] = 16000
    return samples


def synth_silence() -> array:
    """0.5 s of all-zero silence."""
    n_samples = round(SAMPLE_RATE * 0.5)
    return array("h", [0] * n_samples)


def synth_sine_1k_clipped_2x() -> array:
    """1.0 s, 1 kHz sine at 2.0x full scale with half-sample phase offset, hard-clipped."""
    n_samples = round(SAMPLE_RATE * 1.0)
    two_pi_f_over_r = 2.0 * math.pi * 1000.0 / SAMPLE_RATE
    return array(
        "h",
        (quantize_sample(2.0 * math.sin(two_pi_f_over_r * (n + 0.5))) for n in range(n_samples)),
    )


def synthesize_all() -> dict[str, array]:
    """Return a mapping of fixed filename to synthesized sample array for all five files."""
    return {
        "sine_1k.wav": synth_sine_1k(),
        "sweep_log_20_20k.wav": synth_sweep_log_20_20k(),
        "clicks_10.wav": synth_clicks_10(),
        "silence.wav": synth_silence(),
        "sine_1k_clipped_2x.wav": synth_sine_1k_clipped_2x(),
    }
