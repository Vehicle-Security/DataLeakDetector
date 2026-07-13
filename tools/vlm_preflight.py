"""Verify the configured VLM endpoint with one minimal real image request."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "main"))

from data_leak_detector.frame_analyzer.config import VisionConfig
from data_leak_detector.frame_analyzer.frames import KeyFrame
from data_leak_detector.frame_analyzer.vlm_client import VlmRequestFrame
from data_leak_detector.frame_analyzer.vlm_dispatch import build_vlm_clients


def main() -> int:
    config = VisionConfig.from_env()
    if config.vlm_dry_run:
        print("VLM preflight failed: DLD_VLM_DRY_RUN is enabled", file=sys.stderr)
        return 1

    clients = build_vlm_clients(config)
    if not clients or not clients[0].config.vlm_api_key:
        print("VLM preflight failed: no enabled endpoint/API key pair", file=sys.stderr)
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("VLM preflight failed: pillow is not installed", file=sys.stderr)
        return 1

    client = clients[0]
    try:
        with tempfile.TemporaryDirectory(prefix="dld_vlm_preflight_") as temp_dir:
            image_path = Path(temp_dir) / "preflight.jpg"
            Image.new("RGB", (32, 32), color=(245, 245, 245)).save(image_path, quality=80)
            frame = VlmRequestFrame(
                frame=KeyFrame(
                    frame_id="preflight_0",
                    timestamp_ms=0,
                    image_path=str(image_path),
                    score=0.0,
                    reason="preflight",
                    window_id="preflight",
                ),
                visual_note="Minimal endpoint health check.",
                visual_confidence=0.0,
            )
            response = client.analyze([frame], sensitive_files=[], active_apps=[])
    except Exception as exc:
        print(f"VLM preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        "VLM preflight passed: "
        f"model={response.model} endpoint={client.config.vlm_base_url.rstrip('/')} workers={config.vlm_workers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
