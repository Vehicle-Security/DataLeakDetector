# -*- coding: utf-8 -*-
"""
json_io.py - 统一 JSON 文件读写辅助函数

目标：
- 避免 logs/keyevents 在写入过程中被读到半成品
- 为 ETW 输出提供常见 Windows 编码兜底
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_TEXT_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "gb18030",
    "cp936",
    "mbcs",
)


def read_text_with_fallback(path: str, encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS) -> str:
    """按编码优先级读取文本，全部失败时使用 replacement 模式兜底。"""
    raw = Path(path).read_bytes()
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def load_json_file(
    path: str,
    *,
    default: Optional[Any] = None,
    encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS,
) -> Any:
    """读取 JSON 文件；不存在或空文件时返回 default。"""
    if not os.path.exists(path):
        return default

    text = read_text_with_fallback(path, encodings=encodings).strip()
    if not text:
        return default

    return json.loads(text)


def atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """原子写入文本文件，避免读到半写状态。"""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=os.path.splitext(path)[1] or ".tmp",
        dir=directory,
        text=False,
    )

    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """原子写入 JSON 文件。"""
    content = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_text(path, content, encoding="utf-8")
