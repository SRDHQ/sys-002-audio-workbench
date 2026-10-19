#!/usr/bin/env python3
"""Independent closed-form oracle for the audio workbench (M04, D-043, D-045).

Imports ONLY `math` (AC8). Holds its own copy of the parameter table (not shared
with `synth.py`) and computes all expected values from the closed-form formulas
in the Architect specification (revision 1).
"""
import math

SAMPLE_RATE = 48000
CHANNELS = 1
SAMPWIDTH = 2
COMPTYPE = "NONE"
FULL_SCALE = 32767

FILE_NAMES = (
    "sine_1k.wav",
    "sweep_log_20_20k.wav",
    "clicks_10.wav",
    "silence.wav",
    "sine_1k_clipped_2x.wav",
)

DURATIONS = {
    "sine_1k.wav": 1.0,
    "sweep_log_20_20k.wav": 2.0,
    "clicks_10.wav": 1.0,
    "silence.wav": 0.5,
    "sine_1k_clipped_2x.wav": 1.0,
}

# Parameter table (owned independently by oracle.py)
SINE_FREQ_HZ = 1000.0
SINE_AMP = 0.5
SINE_FREQ_TOL_HZ = 0.01

CLIPPED_AMP = 2.0

CLICK_COUNT = 10
CLICK_START = 2400
CLICK_STEP = 4800
CLICK_VALUE = 16000

SWEEP_F1_HZ = 20.0
SWEEP_F2_HZ = 20000.0
SWEEP_DURATION_S = 2.0
SWEEP_START_WINDOW = (0, 9599)
SWEEP_END_WINDOW = (95520, 95999)
SWEEP_REL_TOL = 0.005  # +/-0.5 % relative


def expected_sample_count(filename: str) -> int:
    """Return exact expected sample count N = round(48000 * duration)."""
    return round(SAMPLE_RATE * DURATIONS[filename])


def expected_sine_peak() -> tuple[int, int]:
    """Closed-form max and min integer peak for sine_1k.wav.

    Sample phases are 2*pi*(j + 0.5)/48, so the nearest samples to the positive
    and negative extrema sit at +/- pi/48 radians from the crest/trough.
    """
    max_val = round(FULL_SCALE * SINE_AMP * math.cos(math.pi / 48.0))
    return (max_val, -max_val)


def expected_sine_crossings() -> list[float]:
    """Closed-form upward zero-crossing positions for 1.0 s of 1 kHz sine with +0.5 sample offset."""
    n_samples = expected_sample_count("sine_1k.wav")
    period_samples = SAMPLE_RATE // int(SINE_FREQ_HZ)  # 48
    num_periods = n_samples // period_samples  # 1000
    # Upward crossings occur at x = 47.5 + 48*m for m = 0 .. 998 (999 crossings inside [0, 47999]).
    return [47.5 + period_samples * m for m in range(num_periods - 1)]


def expected_sine_freq() -> tuple[int, float, float]:
    """Return (expected_K, expected_freq_hz, abs_tol_hz) for sine_1k.wav."""
    xs = expected_sine_crossings()
    k = len(xs)
    f_hat = SAMPLE_RATE * (k - 1) / (xs[-1] - xs[0])
    return (k, f_hat, SINE_FREQ_TOL_HZ)


def expected_clipped_count() -> int:
    """Closed-form clipped sample count for sine_1k_clipped_2x.wav (AC5).

    Computed as periods * samples_per_period with |2 * sin(theta_j)| >= 1.0
    from the single-period phase list (48 samples per period, 1000 periods).
    """
    n_samples = expected_sample_count("sine_1k_clipped_2x.wav")
    period_samples = SAMPLE_RATE // int(SINE_FREQ_HZ)  # 48
    num_periods = n_samples // period_samples  # 1000
    clipped_per_period = sum(
        1
        for j in range(period_samples)
        if abs(CLIPPED_AMP * math.sin(2.0 * math.pi * (j + 0.5) / period_samples)) >= 1.0
    )
    return num_periods * clipped_per_period


def expected_largest_non_rail_clipped() -> int:
    """Closed-form largest non-rail |s| in sine_1k_clipped_2x.wav (reported, not a pass condition)."""
    period_samples = SAMPLE_RATE // int(SINE_FREQ_HZ)
    non_rail = [
        round(FULL_SCALE * abs(CLIPPED_AMP * math.sin(2.0 * math.pi * (j + 0.5) / period_samples)))
        for j in range(period_samples)
        if abs(CLIPPED_AMP * math.sin(2.0 * math.pi * (j + 0.5) / period_samples)) < 1.0
    ]
    return max(non_rail)


def expected_rail_count(filename: str) -> int:
    """Expected count of samples with |s| == 32767 for a given file."""
    if filename == "sine_1k_clipped_2x.wav":
        return expected_clipped_count()
    return 0


def expected_clicks() -> list[tuple[int, int]]:
    """Return list of (position, value) pairs for clicks_10.wav."""
    return [(CLICK_START + CLICK_STEP * k, CLICK_VALUE) for k in range(CLICK_COUNT)]


def expected_sweep_window(w_start: int, w_end: int) -> tuple[int, float, float]:
    """Predict upward crossings and mean frequency for sweep_log_20_20k.wav over [w_start, w_end].

    Phase formula: phi(t) = 2*pi * (f1 / c) * (exp(c * t) - 1), where c = ln(f2 / f1) / T.
    Upward crossings occur at phi(t_m) = 2*pi * m for m = 1, 2, ..., so:
      t_m = ln(1 + m * c / f1) / c
      x_m = SAMPLE_RATE * t_m
    Keeps crossings whose sample interval [floor(x_m), floor(x_m) + 1] lies inside [w_start, w_end].
    Returns (expected_K, expected_f_hat, rel_tol).
    """
    c = math.log(SWEEP_F2_HZ / SWEEP_F1_HZ) / SWEEP_DURATION_S
    xs: list[float] = []
    m = 1
    while True:
        t_m = math.log(1.0 + m * c / SWEEP_F1_HZ) / c
        x_m = SAMPLE_RATE * t_m
        fl = math.floor(x_m)
        if fl + 1 > w_end:
            break
        if fl >= w_start:
            xs.append(x_m)
        m += 1
    k = len(xs)
    f_hat = SAMPLE_RATE * (k - 1) / (xs[-1] - xs[0])
    return (k, f_hat, SWEEP_REL_TOL)


def expected_sweep_start() -> tuple[int, float, float]:
    """Return (expected_K, expected_f_hat, rel_tol) for the sweep start window 0..9599."""
    return expected_sweep_window(*SWEEP_START_WINDOW)


def expected_sweep_end() -> tuple[int, float, float]:
    """Return (expected_K, expected_f_hat, rel_tol) for the sweep end window 95520..95999."""
    return expected_sweep_window(*SWEEP_END_WINDOW)


def expected_table() -> dict[str, object]:
    """Return a complete dictionary of closed-form expected values for the five files."""
    sine_k, sine_f, sine_tol = expected_sine_freq()
    sine_max, sine_min = expected_sine_peak()
    sw_start_k, sw_start_f, sw_start_tol = expected_sweep_start()
    sw_end_k, sw_end_f, sw_end_tol = expected_sweep_end()
    return {
        "files": FILE_NAMES,
        "format": {
            "nchannels": CHANNELS,
            "sampwidth": SAMPWIDTH,
            "framerate": SAMPLE_RATE,
            "comptype": COMPTYPE,
        },
        "counts": {name: expected_sample_count(name) for name in FILE_NAMES},
        "rails": {name: expected_rail_count(name) for name in FILE_NAMES},
        "neg32768": {name: 0 for name in FILE_NAMES},
        "sine": {
            "k": sine_k,
            "freq_hz": sine_f,
            "freq_tol_hz": sine_tol,
            "peak_max": sine_max,
            "peak_min": sine_min,
        },
        "clip": {
            "count": expected_clipped_count(),
            "largest_non_rail": expected_largest_non_rail_clipped(),
        },
        "clicks": {
            "positions": [pos for pos, _ in expected_clicks()],
            "values": [val for _, val in expected_clicks()],
            "pairs": expected_clicks(),
        },
        "silence": {
            "nonzero_count": 0,
        },
        "sweep": {
            "start_window": SWEEP_START_WINDOW,
            "start_k": sw_start_k,
            "start_freq_hz": sw_start_f,
            "start_rel_tol": sw_start_tol,
            "end_window": SWEEP_END_WINDOW,
            "end_k": sw_end_k,
            "end_freq_hz": sw_end_f,
            "end_rel_tol": sw_end_tol,
        },
    }
