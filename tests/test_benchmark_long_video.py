"""Focused tests for OSS publication metadata used by the Figure-2 runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_long_video_test_module", ROOT / "tools" / "benchmark_long_video.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class _FakeBucket:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, dict[str, str]]] = []

    def put_object_from_file(self, key: str, filename: str, *, headers: dict[str, str]) -> None:
        self.uploads.append((key, filename, headers))

    def sign_url(self, method: str, key: str, expires: int, *, slash_safe: bool) -> str:
        assert (method, expires, slash_safe) == ("GET", 60, True)
        return f"https://example.oss-cn-beijing.aliyuncs.com/{key}?x-oss-signature=secret"


def test_oss_publisher_uses_isolated_prefix_and_never_persists_signed_url(tmp_path: Path) -> None:
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"synthetic test video")
    prepared = runner.PreparedInput(
        scenario="A", case_dir="/case", duration_min=10, video_file=str(video), log_file="/logs.json",
        video_duration_seconds=600.0, prefix_log_records=1, recording_start_ms=0,
    )
    bucket = _FakeBucket()
    publisher = runner.OssPublisher(bucket, bucket_name="bucket", object_prefix="direct-benchmark/", url_expires_seconds=60)

    entry = publisher.publish(prepared)

    assert entry["publisher"] == "oss"
    assert entry["object_key"].startswith("direct-benchmark/A/10m-")
    assert "url" not in entry
    assert bucket.uploads[0][0] == entry["object_key"]
    url, reference = runner._resolve_direct_url(entry, publisher)
    assert url.endswith("?x-oss-signature=secret")
    assert reference == f"oss://bucket/{entry['object_key']}"


def test_signed_urls_are_redacted_from_persisted_payloads() -> None:
    payload = {"message": "download https://bucket.oss.example/video.mp4?signature=secret"}

    assert runner._redact_payload_urls(payload) == {
        "message": "download https://bucket.oss.example/video.mp4?<redacted>"
    }
