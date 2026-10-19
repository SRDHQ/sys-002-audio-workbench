#!/usr/bin/env python3
"""CLI generator for the five audio workbench test signals (M04, D-043, D-045).

Usage:
  python -I projects/audio-workbench/generate.py --out <dir>

Safety & no-clobber guarantees (AC1):
- Resolves `<dir>` and refuses with exit 2 if it is inside the repository
  (checked on both absolute and symlink-resolved paths).
- Creates `<dir>` if missing (`parents=True`); refuses with exit 2 if `<dir>`
  exists and is not a directory.
- Refuses with exit 2 before writing anything if any of the five target files
  already exists in `<dir>`.
- Writes each WAV to a unique temporary file opened with mode `"xb"` in `<dir>`,
  publishes it atomically via `os.link(tmp, final)`, and always unlinks the
  temp file. On `FileExistsError` or any other I/O error during publishing,
  cleans up temp files, lists any already-published files on stderr, and exits 1.
"""
from __future__ import annotations

import argparse
from array import array
import os
from pathlib import Path
import sys
import uuid
import wave

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import synth  # type: ignore[import-not-found]
else:
    from . import synth


def repo_root() -> Path:
    """Return the repository root `Path(__file__).resolve().parents[2]`."""
    return Path(__file__).resolve().parents[2]


def check_out_dir(out_dir: str | Path, root: Path | None = None) -> Path:
    """Validate `<dir>` is outside the repository on both absolute and resolved paths."""
    base_resolved = (root or repo_root()).resolve()
    base_abs = Path(os.path.abspath(str(root or repo_root())))

    raw_path = Path(out_dir).expanduser()
    abs_path = Path(os.path.abspath(str(raw_path)))
    if abs_path == base_abs or base_abs in abs_path.parents or abs_path == base_resolved or base_resolved in abs_path.parents:
        raise ValueError(f"--out directory {abs_path} is inside repository {base_resolved}")

    resolved = raw_path.resolve()
    if resolved == base_resolved or base_resolved in resolved.parents or resolved == base_abs or base_abs in resolved.parents:
        raise ValueError(f"--out directory {resolved} resolves inside repository {base_resolved}")

    return resolved


def write_wav_atomic(target_path: Path, samples: array) -> None:
    """Write `samples` as 48 kHz 16-bit mono PCM WAV to a `"xb"` temp file and `os.link` to `target_path`."""
    parent = target_path.parent
    tmp_path = parent / f".tmp-{target_path.name}-{os.getpid()}-{uuid.uuid4().hex}.wav"
    try:
        with open(tmp_path, "xb") as fobj:
            with wave.open(fobj, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(synth.SAMPLE_RATE)
                wf.setnframes(len(samples))
                wf.writeframes(samples.tobytes())
            fobj.flush()
            os.fsync(fobj.fileno())
        os.link(tmp_path, target_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def generate_files(out_dir: str | Path, *, root: Path | None = None) -> tuple[int, list[Path]]:
    """Validate `out_dir` and generate all five WAV files.

    Returns `(exit_code, published_paths)`.
    Exit codes:
      0 - all five files written
      2 - usage / path refusal / pre-existing target file (nothing written)
      1 - I/O or atomic link collision during writing
    """
    try:
        resolved_dir = check_out_dir(out_dir, root=root)
    except ValueError as exc:
        print(f"generate.py error: {exc}", file=sys.stderr)
        return (2, [])

    if resolved_dir.exists() and not resolved_dir.is_dir():
        print(f"generate.py error: --out path exists and is not a directory: {resolved_dir}", file=sys.stderr)
        return (2, [])

    try:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"generate.py error: cannot create output directory {resolved_dir}: {exc}", file=sys.stderr)
        return (2, [])

    # Never overwrite: check all five target names before writing anything.
    existing = [name for name in synth.FILE_NAMES if (resolved_dir / name).exists() or (resolved_dir / name).is_symlink()]
    if existing:
        print(
            f"generate.py error: refusing to overwrite existing file(s) in {resolved_dir}: {', '.join(existing)}",
            file=sys.stderr,
        )
        return (2, [])

    signals = synth.synthesize_all()
    published: list[Path] = []
    try:
        for name in synth.FILE_NAMES:
            target = resolved_dir / name
            write_wav_atomic(target, signals[name])
            published.append(target)
    except (FileExistsError, OSError) as exc:
        print(f"generate.py error: failed while writing {name}: {exc}", file=sys.stderr)
        if published:
            print(
                "generate.py error: files already published before failure: "
                + ", ".join(str(p) for p in published),
                file=sys.stderr,
            )
        return (1, published)

    for p in published:
        print(str(p))
    return (0, published)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the five audio workbench WAV signals.")
    ap.add_argument("--out", required=True, help="Output directory outside the repository")
    args = ap.parse_args(argv)
    code, _ = generate_files(args.out)
    return code


if __name__ == "__main__":
    sys.exit(main())
