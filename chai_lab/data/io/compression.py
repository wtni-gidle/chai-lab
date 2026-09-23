# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Small, filename-independent helpers for text artifacts."""

import gzip
import io
import lzma
import os
import uuid
from pathlib import Path

_GZIP_MAGIC = b"\x1f\x8b"
_XZ_MAGIC = b"\xfd7zXZ\x00"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _is_zstd(data: bytes) -> bool:
    if data.startswith(_ZSTD_MAGIC):
        return True
    if len(data) < 4:
        return False
    magic = int.from_bytes(data[:4], "little")
    return 0x184D2A50 <= magic <= 0x184D2A5F  # zstd skippable frames


def _zstd_decompress(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        import zstandard as zstd

        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
            return reader.read()
    return zstd.decompress(data)


def _zstd_compress(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        import zstandard as zstd

        return zstd.ZstdCompressor().compress(data)
    return zstd.compress(data)


def read_text_auto(path: str | Path) -> str:
    """Read UTF-8 text, detecting gzip, xz, or zstd from content magic bytes."""
    path = Path(path)
    data = path.read_bytes()
    try:
        if data.startswith(_GZIP_MAGIC):
            data = gzip.decompress(data)
        elif data.startswith(_XZ_MAGIC):
            data = lzma.decompress(data)
        elif _is_zstd(data):
            data = _zstd_decompress(data)
        return data.decode("utf-8")
    except Exception as error:
        raise ValueError(f"Cannot decode text artifact {path}: {error}") from error


def write_zstd_text(path: str | Path, text: str, *, compress: bool = True) -> Path:
    """Atomically write UTF-8 text, optionally as zstd, and return its path."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        data = text.encode("utf-8")
        temporary_path.write_bytes(_zstd_compress(data) if compress else data)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
