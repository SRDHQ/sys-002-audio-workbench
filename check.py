#!/usr/bin/env python3
"""CLI check program for the audio workbench (M04, D-043, D-045).

Usage:
  python -I check.py <dir>

Compares `measure` results against `oracle` closed-form expectations without
importing or calling `synth` or `generate` (AC8). Also cross-checks the `wave`-based
reader (`measure`) against the `struct`-based reader (`wavparse`) per file (`xread.*`, AC12).
Prints one line per check:
  `<check-id> PASS|FAIL measured=<v> expected=<v> tol=<v>`
followed by:
  `summary: <passed>/<total>`

Exit codes:
  0 - every check passed
  1 - one or more checks failed (including missing/corrupt/wrong-format files)
  2 - usage error (e.g. `<dir>` does not exist or is not a directory)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import measure  # type: ignore[import-not-found]
    import oracle  # type: ignore[import-not-found]
    import wavparse  # type: ignore[import-not-found]
else:
    from . import measure, oracle, wavparse


@dataclass(frozen=True)
class CheckResult:
    """Result of a single named check."""

    check_id: str
    passed: bool
    measured: str
    expected: str
    tol: str

    def format_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.check_id} {status} measured={self.measured} expected={self.expected} tol={self.tol}"


def _check_common_file(filename: str, m: measure.WavMeasurement) -> list[CheckResult]:
    """Run fmt.<file>, count.<file>, rail.<file>, and neg32768.<file> checks."""
    results: list[CheckResult] = []
    exp_n = oracle.expected_sample_count(filename)
    exp_rail = oracle.expected_rail_count(filename)

    # 1. fmt.<file>
    exp_fmt = "1ch_16bit_48000Hz_NONE"
    if m.error == "missing":
        results.append(CheckResult(f"fmt.{filename}", False, "error:missing", exp_fmt, "exact"))
    elif m.error and m.error.startswith("read_error:"):
        results.append(CheckResult(f"fmt.{filename}", False, f"error:{m.error}", exp_fmt, "exact"))
    else:
        meas_fmt = f"{m.nchannels}ch_{m.sampwidth * 8}bit_{m.framerate}Hz_{m.comptype}"
        fmt_pass = (
            m.nchannels == oracle.CHANNELS
            and m.sampwidth == oracle.SAMPWIDTH
            and m.framerate == oracle.SAMPLE_RATE
            and m.comptype == oracle.COMPTYPE
        )
        results.append(CheckResult(f"fmt.{filename}", fmt_pass, meas_fmt, exp_fmt, "exact"))

    # 2. count.<file>
    exp_count_str = f"{exp_n}frames_{2 * exp_n}B"
    if not m.ok:
        results.append(CheckResult(f"count.{filename}", False, f"error:{m.error}", exp_count_str, "exact"))
    else:
        meas_count_str = f"{m.nframes_header}frames_{m.raw_bytes_len}B"
        count_pass = m.nframes_header == exp_n and m.raw_bytes_len == 2 * exp_n
        results.append(CheckResult(f"count.{filename}", count_pass, meas_count_str, exp_count_str, "exact"))

    # 3. rail.<file>
    if not m.ok:
        results.append(CheckResult(f"rail.{filename}", False, f"error:{m.error}", str(exp_rail), "0"))
    else:
        rail_pass = m.rail_count == exp_rail
        results.append(CheckResult(f"rail.{filename}", rail_pass, str(m.rail_count), str(exp_rail), "0"))

    # 4. neg32768.<file>
    if not m.ok:
        results.append(CheckResult(f"neg32768.{filename}", False, f"error:{m.error}", "0", "0"))
    else:
        neg_pass = m.neg32768_count == 0
        results.append(CheckResult(f"neg32768.{filename}", neg_pass, str(m.neg32768_count), "0", "0"))

    return results


def _check_sine(m: measure.WavMeasurement) -> list[CheckResult]:
    """Run sine.freq and sine.peak checks on sine_1k.wav."""
    results: list[CheckResult] = []
    exp_k, exp_f, tol_f = oracle.expected_sine_freq()
    exp_max, exp_min = oracle.expected_sine_peak()

    exp_freq_str = f"K={exp_k},f={exp_f:.6f}Hz"
    tol_freq_str = f"+/-{tol_f}Hz"
    if not m.ok or m.samples is None:
        results.append(CheckResult("sine.freq", False, f"error:{m.error}", exp_freq_str, tol_freq_str))
    else:
        try:
            k, f_hat = measure.estimate_frequency(m.samples, sample_rate=oracle.SAMPLE_RATE)
            freq_pass = (k == exp_k) and (abs(f_hat - exp_f) <= tol_f)
            meas_freq_str = f"K={k},f={f_hat:.6f}Hz"
        except ValueError as exc:
            freq_pass = False
            meas_freq_str = f"error:{str(exc).replace(' ', '_')}"
        results.append(CheckResult("sine.freq", freq_pass, meas_freq_str, exp_freq_str, tol_freq_str))

    exp_peak_str = f"max={exp_max},min={exp_min}"
    if not m.ok or m.samples is None or len(m.samples) == 0:
        err = m.error or "empty_samples"
        results.append(CheckResult("sine.peak", False, f"error:{err}", exp_peak_str, "0"))
    else:
        s_max, s_min = measure.measure_peaks(m.samples)
        peak_pass = (s_max == exp_max) and (s_min == exp_min)
        results.append(CheckResult("sine.peak", peak_pass, f"max={s_max},min={s_min}", exp_peak_str, "0"))

    return results


def _check_clipped(m: measure.WavMeasurement) -> list[CheckResult]:
    """Run clip.count check on sine_1k_clipped_2x.wav (also reports largest non-rail |s|)."""
    exp_clip = oracle.expected_clipped_count()
    if not m.ok or m.samples is None:
        return [CheckResult("clip.count", False, f"error:{m.error}", str(exp_clip), "0")]

    pos_rail = sum(1 for s in m.samples if s == oracle.FULL_SCALE)
    neg_rail = sum(1 for s in m.samples if s == -oracle.FULL_SCALE)
    total_rail = pos_rail + neg_rail
    largest_non_rail = measure.measure_largest_non_rail(m.samples)
    clip_pass = (total_rail == exp_clip) and (pos_rail == exp_clip // 2) and (neg_rail == exp_clip // 2)
    meas_str = f"{total_rail}(+{pos_rail}/-{neg_rail},max_non_rail={largest_non_rail})"
    exp_str = f"{exp_clip}(+{exp_clip // 2}/-{exp_clip // 2})"
    return [CheckResult("clip.count", clip_pass, meas_str, exp_str, "0")]


def _check_clicks(m: measure.WavMeasurement) -> list[CheckResult]:
    """Run clicks.positions check on clicks_10.wav."""
    exp_pairs = oracle.expected_clicks()
    exp_positions = [p for p, _ in exp_pairs]
    exp_str = f"count={len(exp_pairs)},pos={','.join(str(p) for p in exp_positions)},val={oracle.CLICK_VALUE}"
    if not m.ok or m.samples is None:
        return [CheckResult("clicks.positions", False, f"error:{m.error}", exp_str, "exact")]

    nonzero = measure.measure_nonzero(m.samples)
    clicks_pass = nonzero == exp_pairs
    if clicks_pass:
        meas_str = exp_str
    else:
        pos_preview = ",".join(str(p) for p, _ in nonzero[:12])
        if len(nonzero) > 12:
            pos_preview += ",..."
        vals = sorted({v for _, v in nonzero})
        val_str = ",".join(str(v) for v in vals[:5]) if vals else "none"
        meas_str = f"count={len(nonzero)},pos={pos_preview or 'none'},val={val_str}"
    return [CheckResult("clicks.positions", clicks_pass, meas_str, exp_str, "exact")]


def _check_silence(m: measure.WavMeasurement) -> list[CheckResult]:
    """Run silence.zeros check on silence.wav."""
    if not m.ok or m.samples is None:
        return [CheckResult("silence.zeros", False, f"error:{m.error}", "0", "0")]

    nonzero_count = sum(1 for s in m.samples if s != 0)
    return [CheckResult("silence.zeros", nonzero_count == 0, str(nonzero_count), "0", "0")]


def _check_sweep_endpoint(
    check_id: str,
    m: measure.WavMeasurement,
    window: tuple[int, int],
    exp_tuple: tuple[int, float, float],
) -> CheckResult:
    """Run sweep.start or sweep.end check over `window`."""
    exp_k, exp_f, rel_tol = exp_tuple
    exp_str = f"K={exp_k},f={exp_f:.6f}Hz"
    tol_str = f"+/-{rel_tol * 100:.1f}%"
    if not m.ok or m.samples is None:
        return CheckResult(check_id, False, f"error:{m.error}", exp_str, tol_str)

    w_start, w_end = window
    if len(m.samples) <= w_end:
        return CheckResult(check_id, False, f"error:short_file({len(m.samples)})", exp_str, tol_str)

    try:
        k, f_hat = measure.estimate_frequency(
            m.samples,
            sample_rate=oracle.SAMPLE_RATE,
            w_start=w_start,
            w_end=w_end,
        )
        rel_err = abs(f_hat - exp_f) / exp_f
        passed = (k == exp_k) and (rel_err <= rel_tol)
        meas_str = f"K={k},f={f_hat:.6f}Hz"
    except ValueError as exc:
        passed = False
        meas_str = f"error:{str(exc).replace(' ', '_')}"

    return CheckResult(check_id, passed, meas_str, exp_str, tol_str)


XREAD_MAX_BYTES = 1_048_576


def compare_readers(parsed: wavparse.ParsedWav, m: measure.WavMeasurement) -> str:
    """Compare the `struct` parse with the `wave` measurement; return `agree`, `diff:...` or `error:...`.

    Stops at the first difference, in a fixed field order. Does not compare with the oracle.
    """
    if parsed.samples is None:
        return "error:struct:not_16bit_mono"
    pairs = (
        ("channels", m.nchannels, parsed.channels),
        ("sampwidth", m.sampwidth, parsed.bits // 8),
        ("framerate", m.framerate, parsed.sample_rate),
        ("nframes", m.nframes_header, parsed.nframes),
        ("raw_bytes", m.raw_bytes_len, 2 * len(parsed.samples)),
    )
    for field, wave_v, struct_v in pairs:
        if wave_v != struct_v:
            return f"diff:{field}(wave={wave_v},struct={struct_v})"
    if m.samples is None:
        return "error:wave:no_samples"
    differing = [i for i, (a, b) in enumerate(zip(m.samples, parsed.samples)) if a != b]
    if differing:
        return f"diff:samples(first={differing[0]},count={len(differing)})"
    return "agree"


def _check_xread(filename: str, path: Path, m: measure.WavMeasurement) -> CheckResult:
    """Run xread.<file>: both readers must agree. Every failure is a FAIL with a reason token."""
    check_id = f"xread.{filename}"

    def result(measured: str) -> CheckResult:
        return CheckResult(check_id, measured == "agree", measured, "agree", "exact")

    try:
        with path.open("rb") as fobj:
            data = fobj.read(XREAD_MAX_BYTES + 1)
    except FileNotFoundError:
        return result("error:missing")
    except OSError as exc:
        return result(f"error:struct:read_error:{type(exc).__name__}")
    if len(data) > XREAD_MAX_BYTES:
        return result("error:struct:too_large")
    try:
        parsed = wavparse.parse_wav(data)
    except wavparse.WavFormatError as exc:
        return result(f"error:struct:{exc.reason}")
    if parsed.chunk_ids != ("fmt ", "data"):
        return result(f"error:struct:layout:{'+'.join(parsed.chunk_ids)}")
    if not m.ok:
        return result(f"error:wave:{m.error}")
    return result(compare_readers(parsed, m))


def run_all_checks(target_dir: str | Path) -> list[CheckResult]:
    """Measure all five WAV files in `target_dir` and return all CheckResults in deterministic order."""
    base = Path(target_dir)
    measurements = {name: measure.read_and_measure_wav(base / name) for name in oracle.FILE_NAMES}

    results: list[CheckResult] = []
    for name in oracle.FILE_NAMES:
        results.extend(_check_common_file(name, measurements[name]))

    results.extend(_check_sine(measurements["sine_1k.wav"]))
    results.extend(_check_clipped(measurements["sine_1k_clipped_2x.wav"]))
    results.extend(_check_clicks(measurements["clicks_10.wav"]))
    results.extend(_check_silence(measurements["silence.wav"]))
    results.append(
        _check_sweep_endpoint(
            "sweep.start",
            measurements["sweep_log_20_20k.wav"],
            oracle.SWEEP_START_WINDOW,
            oracle.expected_sweep_start(),
        )
    )
    results.append(
        _check_sweep_endpoint(
            "sweep.end",
            measurements["sweep_log_20_20k.wav"],
            oracle.SWEEP_END_WINDOW,
            oracle.expected_sweep_end(),
        )
    )
    for name in oracle.FILE_NAMES:
        results.append(_check_xread(name, base / name, measurements[name]))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check the five audio workbench WAV signals against the oracle.")
    ap.add_argument("dir", help="Directory containing the five WAV files")
    args = ap.parse_args(argv)

    target_dir = Path(args.dir).expanduser()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"check.py error: directory not found: {args.dir}", file=sys.stderr)
        return 2

    results = run_all_checks(target_dir)
    for r in results:
        print(r.format_line())

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    print(f"summary: {passed_count}/{total_count}")
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
