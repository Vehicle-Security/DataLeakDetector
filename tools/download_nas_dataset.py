"""Download NAS dataset files for local full-suite testing.

The Synology FileStation API sometimes reports zero file sizes for this NAS, so
the downloader relies on local completed files instead of remote size metadata.
Interrupted downloads are written as ``.part`` files and retried on the next run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_BASE_URL = "https://iot408.cn3.quickconnect.cn"
DEFAULT_ROOTS = ["/video/stage1", "/video/stage2", "/video/stage4", "/video/stage5"]
USEFUL_FILE_NAMES = {
    "INDEX.md",
    "groundtruth.json",
    "groungtruth.json",
    "groudtruth.json",
}
USEFUL_DIR_NAMES = {"logs", "video"}


class FileStationClient:
    def __init__(self, base_url: str, account: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.entry_url = f"{self.base_url}/webapi/entry.cgi"
        self.account = account
        self.password = password
        self.sid = ""

    def __enter__(self) -> "FileStationClient":
        self.login()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.logout()

    def _post(self, data: Dict[str, str], timeout: int = 60, retry: int = 3) -> Dict[str, Any]:
        body = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(self.entry_url, data=body, method="POST")
        last_error: BaseException | None = None
        for attempt in range(1, retry + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    payload = response.read().decode("utf-8")
                result = json.loads(payload)
                if not result.get("success"):
                    raise RuntimeError(f"FileStation API failed: {result}")
                return result
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                if attempt >= retry:
                    break
                time.sleep(min(30, 2 * attempt))
        raise RuntimeError(f"FileStation request failed after {retry} attempts: {last_error}")

    def login(self) -> None:
        result = self._post(
            {
                "api": "SYNO.API.Auth",
                "version": "7",
                "method": "login",
                "account": self.account,
                "passwd": self.password,
                "session": "FileStation",
                "format": "sid",
            }
        )
        self.sid = result["data"]["sid"]

    def logout(self) -> None:
        if not self.sid:
            return
        try:
            self._post(
                {
                    "api": "SYNO.API.Auth",
                    "version": "7",
                    "method": "logout",
                    "session": "FileStation",
                    "_sid": self.sid,
                },
                timeout=20,
            )
        finally:
            self.sid = ""

    def list_dir(self, folder_path: str) -> List[Dict[str, Any]]:
        result = self._post(
            {
                "api": "SYNO.FileStation.List",
                "version": "2",
                "method": "list",
                "folder_path": folder_path,
                "additional": "size,time",
                "limit": "1000",
                "_sid": self.sid,
            }
        )
        return list(result.get("data", {}).get("files", []))

    def download_file(self, remote_path: str, local_path: Path, timeout: int = 1800) -> int:
        params = urllib.parse.urlencode(
            {
                "api": "SYNO.FileStation.Download",
                "version": "2",
                "method": "download",
                "mode": "download",
                "path": remote_path,
                "_sid": self.sid,
            }
        )
        url = f"{self.entry_url}?{params}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = local_path.with_name(local_path.name + ".part")
        if part_path.exists():
            part_path.unlink()
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with part_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        part_path.replace(local_path)
        return local_path.stat().st_size


def should_descend(path: str, include_results: bool) -> bool:
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if name == "results" and not include_results:
        return False
    if name in USEFUL_DIR_NAMES or name.startswith("session_"):
        return True
    return True


def should_download(path: str, include_results: bool) -> bool:
    parts = path.strip("/").split("/")
    name = parts[-1]
    if "results" in parts and not include_results:
        return False
    if name in USEFUL_FILE_NAMES:
        return True
    if any(part in USEFUL_DIR_NAMES for part in parts):
        return True
    return False


def iter_files(client: FileStationClient, roots: Iterable[str], include_results: bool) -> Iterable[str]:
    stack = list(roots)
    while stack:
        folder = stack.pop()
        for item in client.list_dir(folder):
            path = item["path"]
            if item.get("isdir"):
                if should_descend(path, include_results):
                    stack.append(path)
            elif should_download(path, include_results):
                yield path


def local_path_for(remote_path: str, output_dir: Path) -> Path:
    relative = remote_path.strip("/")
    if relative.startswith("video/"):
        relative = relative[len("video/") :]
    return output_dir / relative


def main() -> int:
    parser = argparse.ArgumentParser(description="Download useful NAS dataset files.")
    parser.add_argument("--base-url", default=os.getenv("NAS_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--account", default=os.getenv("NAS_ACCOUNT", "video"))
    parser.add_argument("--password", default=os.getenv("NAS_PASSWORD"))
    parser.add_argument("--output-dir", type=Path, default=Path("spec/data/nas_samples"))
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--include-results", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("spec/data/nas_samples/download_manifest.jsonl"))
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--retry", type=int, default=3)
    args = parser.parse_args()

    if not args.password:
        print("NAS_PASSWORD is required.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    total = skipped = downloaded = failed = 0
    started_at = time.time()

    with FileStationClient(args.base_url, args.account, args.password) as client:
        with args.manifest.open("a", encoding="utf-8") as manifest:
            for remote_path in iter_files(client, args.roots, args.include_results):
                total += 1
                if args.max_files and total > args.max_files:
                    break

                local_path = local_path_for(remote_path, args.output_dir)
                if local_path.exists() and local_path.stat().st_size > 0:
                    skipped += 1
                    print(f"[skip] {remote_path}")
                    continue

                for attempt in range(1, args.retry + 1):
                    try:
                        size = client.download_file(remote_path, local_path)
                        downloaded += 1
                        event = {
                            "remote_path": remote_path,
                            "local_path": str(local_path),
                            "bytes": size,
                            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        manifest.write(json.dumps(event, ensure_ascii=False) + "\n")
                        manifest.flush()
                        print(f"[ok] {remote_path} -> {local_path} ({size} bytes)")
                        break
                    except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                        if attempt >= args.retry:
                            failed += 1
                            print(f"[fail] {remote_path}: {exc}", file=sys.stderr)
                        else:
                            wait_seconds = min(30, 2 * attempt)
                            print(f"[retry {attempt}] {remote_path}: {exc}; waiting {wait_seconds}s")
                            time.sleep(wait_seconds)

    elapsed = time.time() - started_at
    print(
        json.dumps(
            {
                "seen": total,
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 2),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
