#!/usr/bin/env python3
"""Simulated tests for the audio workbench (M04, D-043, D-045).

All test data is simulated (`SIMULATED = True`). Nothing is written outside
`tempfile` directories. Covers all 6 test groups from the Architect specification
(revision 1):
1. Round trip (AC1, AC3-AC7): `generate.py` into a temp dir, then `check.py` passes every check.
2. Refusals (AC1): inside repo (direct & symlink), pre-existing file (bytes unchanged, no other file created), second run refused, atomic `os.link` collision cleanup.
3. The checker can fail: WAVs written directly with `wave`/`array` (without `synth`), each wrong in one way, and the named check must FAIL.
4. Estimator units: `[-2, 2]` -> 0.5; `[-1, 0]` -> 1.0; `[1, 0, -1]` -> no crossing; <2 crossings -> error, never 0 Hz.
5. Oracle table: reproduces the spec's Expected values (exact integers; sweep endpoints within 1e-6 relative; K = 5 and 196).
6. Independence (AC8): `ast` scan of `oracle.py`, `measure.py`, and `check.py` for any import of `synth` or `generate`; `oracle.py` imports only `math`.
"""
from __future__ import annotations

from array import array
import ast
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR  # the project's own repository (M05.5)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import check  # type: ignore[import-not-found]
import generate  # type: ignore[import-not-found]
import measure  # type: ignore[import-not-found]
import oracle  # type: ignore[import-not-found]

SIMULATED = True


def _write_raw_wav(
    path: Path,
    samples: list[int] | array | bytes,
    *,
    nchannels: int = 1,
    sampwidth: int = 2,
    framerate: int = 48000,
) -> None:
    """Simulated test helper: write a WAV file directly with `wave`/`array`, without `synth`."""
    with path.open("wb") as fobj:
        with wave.open(fobj, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            if isinstance(samples, bytes):
                wf.setnframes(len(samples) // (nchannels * sampwidth))
                wf.writeframes(samples)
            else:
                arr = samples if isinstance(samples, array) else array("h", samples)
                wf.setnframes(len(arr) // nchannels)
                wf.writeframes(arr.tobytes())


def _parse_check_output(stdout: str) -> tuple[dict[str, tuple[str, str, str, str]], str]:
    """Parse `check.py` stdout into `{check_id: (status, measured, expected, tol)}` and summary line."""
    checks: dict[str, tuple[str, str, str, str]] = {}
    summary = ""
    for raw_line in stdout.strip().splitlines():
        line = raw_line.strip()
        if line.startswith("summary:"):
            summary = line
            continue
        parts = line.split(" ")
        if len(parts) != 5:
            raise AssertionError(f"unexpected check output line format: {line!r}")
        cid, status, meas, exp, tol = parts
        assert status in ("PASS", "FAIL"), f"unexpected status in {line!r}"
        assert meas.startswith("measured="), f"missing measured= in {line!r}"
        assert exp.startswith("expected="), f"missing expected= in {line!r}"
        assert tol.startswith("tol="), f"missing tol= in {line!r}"
        checks[cid] = (status, meas[len("measured="):], exp[len("expected="):], tol[len("tol="):])
    return checks, summary


class RoundTripTests(unittest.TestCase):
    """Test group 1: Round trip (AC1, AC3-AC7) via `python -I` subprocesses."""

    def test_generate_and_check_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-audio-wb-") as tmp:
            out_dir = Path(tmp) / "nested" / "out"
            gen_proc = subprocess.run(
                [sys.executable, "-I", str(PROJECT_DIR / "generate.py"), "--out", str(out_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(gen_proc.returncode, 0, msg=gen_proc.stderr)
            written_lines = [line.strip() for line in gen_proc.stdout.strip().splitlines() if line.strip()]
            self.assertEqual(len(written_lines), 5)
            for name in oracle.FILE_NAMES:
                self.assertTrue((out_dir / name).is_file(), f"missing {name}")

            chk_proc = subprocess.run(
                [sys.executable, "-I", str(PROJECT_DIR / "check.py"), str(out_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(chk_proc.returncode, 0, msg=chk_proc.stdout + "\n" + chk_proc.stderr)
            self.assertEqual(chk_proc.stderr, "")
            # Verify no path appears in check.py output
            self.assertNotIn(str(out_dir), chk_proc.stdout)

            checks, summary = _parse_check_output(chk_proc.stdout)
            self.assertEqual(summary, "summary: 27/27")
            for cid, (status, _, _, _) in checks.items():
                self.assertEqual(status, "PASS", f"check {cid} did not pass")


class RefusalTests(unittest.TestCase):
    """Test group 2: Refusals and no-clobber guarantees (AC1)."""

    def test_repo_root_is_this_repository(self) -> None:
        """M05.5: the output boundary is this project's own repository, not its parent folder."""
        self.assertEqual(generate.repo_root(), PROJECT_DIR)

    def test_refuses_out_inside_repo_directly_and_via_symlink(self) -> None:
        inside_dir = REPO_ROOT / "inside-test-out"
        self.assertFalse(inside_dir.exists())
        proc = subprocess.run(
            [sys.executable, "-I", str(PROJECT_DIR / "generate.py"), "--out", str(inside_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertFalse(inside_dir.exists())

        # Symlink outside repo pointing inside repo
        with tempfile.TemporaryDirectory(prefix="sim-audio-symlink-") as tmp:
            link_path = Path(tmp) / "link_to_repo"
            link_path.symlink_to(REPO_ROOT)
            target_via_link = link_path / "symlink-out"
            proc_sym = subprocess.run(
                [sys.executable, "-I", str(PROJECT_DIR / "generate.py"), "--out", str(target_via_link)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_sym.returncode, 2)
            self.assertFalse((REPO_ROOT / "symlink-out").exists())

    def test_refuses_when_one_target_exists_and_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-audio-exists-") as tmp:
            out_dir = Path(tmp)
            sentinel_file = out_dir / "clicks_10.wav"
            sentinel_bytes = b"simulated-preexisting-bytes-do-not-clobber"
            sentinel_file.write_bytes(sentinel_bytes)

            proc = subprocess.run(
                [sys.executable, "-I", str(PROJECT_DIR / "generate.py"), "--out", str(out_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(sentinel_file.read_bytes(), sentinel_bytes)
            for name in oracle.FILE_NAMES:
                if name != "clicks_10.wav":
                    self.assertFalse((out_dir / name).exists(), f"{name} should not have been created")

    def test_second_run_into_same_dir_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-audio-second-") as tmp:
            out_dir = Path(tmp)
            code1, _ = generate.generate_files(out_dir)
            self.assertEqual(code1, 0)
            before_bytes = {name: (out_dir / name).read_bytes() for name in oracle.FILE_NAMES}

            code2, published2 = generate.generate_files(out_dir)
            self.assertEqual(code2, 2)
            self.assertEqual(published2, [])
            for name in oracle.FILE_NAMES:
                self.assertEqual((out_dir / name).read_bytes(), before_bytes[name])

    def test_os_link_race_cleans_temp_file_and_reports_published(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-audio-race-") as tmp:
            out_dir = Path(tmp)
            real_link = os.link
            call_count = 0

            def colliding_link(src: str | Path, dst: str | Path) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    # Simulate a concurrent writer creating the 2nd target right before os.link
                    Path(dst).write_bytes(b"concurrent-sentinel")
                return real_link(src, dst)

            with mock.patch("os.link", side_effect=colliding_link):
                code, published = generate.generate_files(out_dir)

            self.assertEqual(code, 1)
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0].name, "sine_1k.wav")
            # Sentinel at second file was not overwritten
            self.assertEqual((out_dir / "sweep_log_20_20k.wav").read_bytes(), b"concurrent-sentinel")
            # No leftover .tmp- files
            leftovers = list(out_dir.glob(".tmp-*"))
            self.assertEqual(leftovers, [])


class CheckerCanFailTests(unittest.TestCase):
    """Test group 3: The checker can fail on synthetic wrong WAVs written without `synth`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._base_tmp = tempfile.TemporaryDirectory(prefix="sim-audio-baseline-")
        cls.base_dir = Path(cls._base_tmp.name)
        code, _ = generate.generate_files(cls.base_dir)
        assert code == 0, "baseline generation failed"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._base_tmp.cleanup()

    def _prepare_dir_with_Valid_files(self, tmp_dir: Path) -> None:
        for name in oracle.FILE_NAMES:
            (tmp_dir / name).write_bytes((self.base_dir / name).read_bytes())

    def test_wrong_sine_frequency_fails_sine_freq(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-sinefreq-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            # Write a 1000.03 Hz sine directly without synth
            n_samples = 48000
            samples = [
                round(32767 * 0.5 * math.sin(2.0 * math.pi * 1000.03 * (n + 0.5) / 48000.0))
                for n in range(n_samples)
            ]
            _write_raw_wav(d / "sine_1k.wav", samples)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["sine.freq"].passed)
            self.assertIn("f=1000.02999", results["sine.freq"].measured)

    def test_wrong_sine_peak_fails_sine_peak(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-sinepeak-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            # Write a 1000 Hz sine whose peak is bumped to 16349
            n_samples = 48000
            samples = [
                round(32767 * 0.5 * math.sin(2.0 * math.pi * 1000.0 * (n + 0.5) / 48000.0))
                for n in range(n_samples)
            ]
            samples[12] = 16349  # positive peak index
            _write_raw_wav(d / "sine_1k.wav", samples)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["sine.peak"].passed)

    def test_wrong_clipped_count_fails_clip_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-clip-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            # 32,001 rail samples directly without synth
            samples = [32767] * 16001 + [-32767] * 16000 + [0] * (48000 - 32001)
            _write_raw_wav(d / "sine_1k_clipped_2x.wav", samples)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["clip.count"].passed)
            self.assertFalse(results["rail.sine_1k_clipped_2x.wav"].passed)

    def test_wrong_click_position_fails_clicks_positions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-clicks-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            samples = [0] * 48000
            for k in range(10):
                pos = 2401 if k == 0 else (2400 + 4800 * k)
                samples[pos] = 16000
            _write_raw_wav(d / "clicks_10.wav", samples)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["clicks.positions"].passed)

    def test_nonzero_silence_sample_fails_silence_zeros(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-silence-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            samples = [0] * 24000
            samples[1234] = 1
            _write_raw_wav(d / "silence.wav", samples)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["silence.zeros"].passed)

    def test_linear_sweep_fails_sweep_start_and_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-sweep-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            # Linear sweep from 20 Hz to 20 kHz over 2.0 s: phi(t) = 2*pi*(f1*t + 0.5*(f2-f1)/T * t^2)
            n_samples = 96000
            f1, f2, dur = 20.0, 20000.0, 2.0
            k_lin = (f2 - f1) / dur
            samples = []
            for n in range(n_samples):
                t = n / 48000.0
                phi = 2.0 * math.pi * (f1 * t + 0.5 * k_lin * t * t)
                samples.append(round(32767 * 0.5 * math.sin(phi)))
            _write_raw_wav(d / "sweep_log_20_20k.wav", samples)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["sweep.start"].passed)
            self.assertFalse(results["sweep.end"].passed)

    def test_header_and_count_errors_fail_fmt_and_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-fmt-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            # 1) 44,100 Hz header on silence.wav
            _write_raw_wav(d / "silence.wav", [0] * 24000, framerate=44100)
            # 2) stereo file on clicks_10.wav
            _write_raw_wav(d / "clicks_10.wav", [0] * (48000 * 2), nchannels=2)
            # 3) 8-bit file on sine_1k_clipped_2x.wav
            _write_raw_wav(d / "sine_1k_clipped_2x.wav", bytes([128] * 48000), sampwidth=1)
            # 4) one sample short on sine_1k.wav
            _write_raw_wav(d / "sine_1k.wav", [0] * 47999)

            results = {r.check_id: r for r in check.run_all_checks(d)}
            self.assertFalse(results["fmt.silence.wav"].passed)
            self.assertFalse(results["fmt.clicks_10.wav"].passed)
            self.assertFalse(results["fmt.sine_1k_clipped_2x.wav"].passed)
            self.assertTrue(results["fmt.sine_1k.wav"].passed)
            self.assertFalse(results["count.sine_1k.wav"].passed)

    def test_missing_file_fails_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sim-fail-missing-") as tmp:
            d = Path(tmp)
            self._prepare_dir_with_Valid_files(d)
            (d / "sine_1k.wav").unlink()

            proc = subprocess.run(
                [sys.executable, "-I", str(PROJECT_DIR / "check.py"), str(d)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertNotIn("Traceback", proc.stdout)
            checks, _ = _parse_check_output(proc.stdout)
            for cid in ("fmt.sine_1k.wav", "count.sine_1k.wav", "rail.sine_1k.wav", "neg32768.sine_1k.wav", "sine.freq", "sine.peak"):
                status, meas, _, _ = checks[cid]
                self.assertEqual(status, "FAIL", f"{cid} should FAIL when file is missing")
                self.assertTrue(meas.startswith("error:"), f"{cid} should report error:<reason>, got {meas!r}")


class EstimatorUnitTests(unittest.TestCase):
    """Test group 4: Upward crossing and frequency estimator unit tests."""

    def test_upward_crossing_interpolation_cases(self) -> None:
        self.assertEqual(measure.upward_crossings([-2, 2]), [0.5])
        self.assertEqual(measure.upward_crossings([-1, 0]), [1.0])
        self.assertEqual(measure.upward_crossings([1, 0, -1]), [])

    def test_fewer_than_two_crossings_raises_error_never_zero_hz(self) -> None:
        with self.assertRaises(ValueError):
            measure.estimate_frequency([1, 0, -1])
        with self.assertRaises(ValueError):
            measure.estimate_frequency([-2, 2])


class OracleTableTests(unittest.TestCase):
    """Test group 5: Oracle reproduces the Architect spec's Expected values."""

    def test_oracle_values_match_spec(self) -> None:
        # Sample counts
        self.assertEqual(oracle.expected_sample_count("sine_1k.wav"), 48000)
        self.assertEqual(oracle.expected_sample_count("sweep_log_20_20k.wav"), 96000)
        self.assertEqual(oracle.expected_sample_count("clicks_10.wav"), 48000)
        self.assertEqual(oracle.expected_sample_count("silence.wav"), 24000)
        self.assertEqual(oracle.expected_sample_count("sine_1k_clipped_2x.wav"), 48000)

        # Sine peak & frequency
        self.assertEqual(oracle.expected_sine_peak(), (16348, -16348))
        sine_k, sine_f, sine_tol = oracle.expected_sine_freq()
        self.assertEqual(sine_k, 999)
        self.assertEqual(sine_f, 1000.0)
        self.assertEqual(sine_tol, 0.01)

        # Clipped count & largest non-rail
        self.assertEqual(oracle.expected_clipped_count(), 32000)
        self.assertEqual(oracle.expected_largest_non_rail_clipped(), 28985)

        # Clicks
        expected_click_positions = [2400, 7200, 12000, 16800, 21600, 26400, 31200, 36000, 40800, 45600]
        self.assertEqual(oracle.expected_clicks(), [(p, 16000) for p in expected_click_positions])

        # Sweep start & end windows
        start_k, start_f, start_tol = oracle.expected_sweep_start()
        self.assertEqual(start_k, 5)
        self.assertEqual(start_tol, 0.005)
        self.assertLessEqual(abs(start_f - 29.830324) / 29.830324, 1e-6)

        end_k, end_f, end_tol = oracle.expected_sweep_end()
        self.assertEqual(end_k, 196)
        self.assertEqual(end_tol, 0.005)
        self.assertLessEqual(abs(end_f - 19658.552) / 19658.552, 1e-6)


class IndependenceTests(unittest.TestCase):
    """Test group 6: AST scan proving oracle/measure/check never import synth or generate (AC8)."""

    FORBIDDEN = frozenset({"synth", "generate"})

    def _scan_imports(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.Call):
                # Check __import__("...") or importlib.import_module("...")
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in ("__import__", "import_module"):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            imported.add(arg.value.lstrip(".").split(".")[0])
        return imported

    def test_oracle_measure_check_do_not_import_synth_or_generate(self) -> None:
        for mod_name in ("oracle.py", "measure.py", "check.py"):
            imported = self._scan_imports(PROJECT_DIR / mod_name)
            forbidden_found = imported & self.FORBIDDEN
            self.assertEqual(
                forbidden_found,
                set(),
                f"{mod_name} must not import {forbidden_found}",
            )

    def test_oracle_imports_only_math(self) -> None:
        imported = self._scan_imports(PROJECT_DIR / "oracle.py")
        self.assertEqual(imported, {"math"})


if __name__ == "__main__":
    unittest.main()
