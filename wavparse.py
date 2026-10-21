#!/usr/bin/env python3
"""Second, independent WAV reader for the audio workbench (revision 2, closes EXP-002 limitation 1).

Parses the RIFF header and 16-bit samples with `struct` only. It never imports `wave`,
`array`, `synth`, `generate`, `measure` or `oracle`, so a fault that `wave` makes
symmetrically on write and read can still show up here. Every field is unpacked with an
explicit little-endian format, so the result does not depend on `sys.byteorder`.

It accepts only the canonical layout this project writes (PCM, `fmt ` of 16 bytes) and
rejects everything else with a `WavFormatError` reason token. Rules 4, 11, 12 and 13
are the checks `wave` skips (RIFF size, bits, block align, byte rate).
"""
import struct
from dataclasses import dataclass


class WavFormatError(Exception):
    """Malformed or unsupported WAV bytes; `reason` is a short token."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ParsedWav:
    """Header fields and (for 16-bit mono) samples of one parsed WAV."""

    riff_size: int
    fmt_size: int
    format_tag: int
    channels: int
    sample_rate: int
    byte_rate: int
    block_align: int
    bits: int
    data_offset: int
    data_size: int
    nframes: int
    chunk_ids: tuple[str, ...]
    samples: tuple[int, ...] | None


def parse_wav(data: bytes) -> ParsedWav:
    """Parse WAV `data` (the whole file's bytes). Pure: no I/O, no partial results."""
    if len(data) < 12:
        raise WavFormatError("short_file")
    if data[0:4] != b"RIFF":
        raise WavFormatError("not_riff")
    if data[8:12] != b"WAVE":
        raise WavFormatError("not_wave")
    (riff_size,) = struct.unpack_from("<L", data, 4)
    if riff_size != len(data) - 8:
        raise WavFormatError("riff_size_mismatch")

    # Walk the chunks; the walk must end exactly at the end of the file.
    chunks: list[tuple[bytes, int, int]] = []  # (id, body offset, body size)
    pos = 12
    while pos < len(data):
        if pos + 8 > len(data):
            raise WavFormatError("chunk_overrun")
        cid, size = struct.unpack_from("<4sL", data, pos)
        body = pos + 8
        end = body + size
        if size % 2:
            end += 1  # pad byte, which must also fit
        if end > len(data):
            raise WavFormatError("chunk_overrun")
        chunks.append((cid, body, size))
        pos = end

    ids = [c[0] for c in chunks]
    if ids.count(b"fmt ") == 0:
        raise WavFormatError("missing_fmt")
    if ids.count(b"data") == 0:
        raise WavFormatError("missing_data")
    if ids.count(b"fmt ") > 1:
        raise WavFormatError("duplicate_fmt")
    if ids.count(b"data") > 1:
        raise WavFormatError("duplicate_data")
    if ids.index(b"fmt ") > ids.index(b"data"):
        raise WavFormatError("data_before_fmt")

    _, fmt_off, fmt_size = chunks[ids.index(b"fmt ")]
    _, data_off, data_size = chunks[ids.index(b"data")]
    if fmt_size != 16:
        raise WavFormatError("fmt_size")
    format_tag, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from("<HHLLHH", data, fmt_off)
    if format_tag != 1:
        raise WavFormatError("format_tag")
    if channels < 1:
        raise WavFormatError("bad_channels")
    if bits not in (8, 16, 24, 32):
        raise WavFormatError("bad_bits")
    if block_align != channels * bits // 8:
        raise WavFormatError("block_align_mismatch")
    if byte_rate != sample_rate * block_align:
        raise WavFormatError("byte_rate_mismatch")
    if data_size % block_align != 0:
        raise WavFormatError("data_size_not_multiple")

    samples = None
    if channels == 1 and bits == 16:
        n = data_size // 2
        samples = struct.unpack_from(f"<{n}h", data, data_off)

    return ParsedWav(
        riff_size=riff_size,
        fmt_size=fmt_size,
        format_tag=format_tag,
        channels=channels,
        sample_rate=sample_rate,
        byte_rate=byte_rate,
        block_align=block_align,
        bits=bits,
        data_offset=data_off,
        data_size=data_size,
        nframes=data_size // block_align,
        chunk_ids=tuple(i.decode("ascii", errors="replace") for i in ids),
        samples=samples,
    )
