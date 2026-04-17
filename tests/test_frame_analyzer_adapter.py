import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "01-FrameAnalyzer"

if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


from frame_analyzer import adapt_legacy_frame_result  # noqa: E402
from frame_analyzer.service import FrameAnalyzerRequest, FrameAnalyzerService  # noqa: E402


class FrameAnalyzerAdapterTests(unittest.TestCase):
    def test_adapt_legacy_frame_result_builds_segments(self):
        legacy_result = {
            "search_range": {
                "start": "2026-03-27 12:31:46",
                "end": "2026-03-27 12:32:17",
            },
            "events": [
                {
                    "app_name": "QQ邮箱",
                    "behavior_category": "直接外发",
                    "operation_type": "邮件附件外发",
                    "original_filename": "part1.xlsx, part2.xlsx",
                    "modified_filename": "未知",
                    "time_range": "2026-03-27 12:31:46 - 2026-03-27 12:32:17",
                    "involved_timestamps": ["2026-03-27 12:31:46", "2026-03-27 12:31:48"],
                    "description": "用户发送了两个附件。",
                }
            ],
        }

        result = adapt_legacy_frame_result(legacy_result)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["primary_resource"], "part1.xlsx")
        self.assertEqual(result["segments"][0]["related_resources"], ["part2.xlsx"])

    def test_service_force_refresh_bypasses_existing_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.dict("os.environ", {"FRAME_ANALYZER_CACHE_DIR": cache_dir}, clear=False):
                service = FrameAnalyzerService()
                request = FrameAnalyzerRequest(
                    video_path="demo.mp4",
                    recording_start_time="2026-03-27 12:31:00",
                    search_start_time="2026-03-27 12:31:10",
                    search_end_time="2026-03-27 12:31:30",
                    target_keywords=["demo"],
                    force_refresh=True,
                )
                cache_path = service._cache_path(request)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    '{"status":"success","segments":[],"analysis_metadata":{"cache_hit":true}}',
                    encoding="utf-8",
                )

                with mock.patch.object(
                    service.adapter,
                    "analyze_with_legacy_backend",
                    return_value={"status": "success", "segments": []},
                ) as backend_mock:
                    result = service.analyze(request)

        self.assertTrue(backend_mock.called)
        self.assertFalse(result["analysis_metadata"]["cache_hit"])
        self.assertTrue(result["analysis_metadata"]["fresh_run_requested"])

    def test_cached_result_uses_current_request_refresh_semantics(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.dict("os.environ", {"FRAME_ANALYZER_CACHE_DIR": cache_dir}, clear=False):
                service = FrameAnalyzerService()
                request = FrameAnalyzerRequest(
                    video_path="cached-demo.mp4",
                    recording_start_time="2026-03-27 12:31:00",
                    search_start_time="2026-03-27 12:31:10",
                    search_end_time="2026-03-27 12:31:30",
                    target_keywords=["demo"],
                    force_refresh=False,
                )
                cache_path = service._cache_path(request)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    (
                        '{"status":"success","segments":[],"analysis_metadata":'
                        '{"fresh_run_requested":true,"request_signature":"old","cache_path":"old-path"}}'
                    ),
                    encoding="utf-8",
                )

                result = service.analyze(request)

        self.assertTrue(result["analysis_metadata"]["cache_hit"])
        self.assertFalse(result["analysis_metadata"]["fresh_run_requested"])
        self.assertEqual(result["analysis_metadata"]["request_signature"], service._request_digest(request))
        self.assertEqual(result["analysis_metadata"]["cache_path"], str(cache_path))
        self.assertNotIn("cached_result_fresh_run_requested", result["analysis_metadata"])

    def test_debug_frame_dir_is_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            from frame_analyzer.legacy_agent import _resolve_debug_frame_dir

            self.assertIsNone(_resolve_debug_frame_dir())

    def test_debug_frame_dir_uses_explicit_env_toggle(self):
        with mock.patch.dict(
            "os.environ",
            {
                "FRAME_ANALYZER_SAVE_DEBUG_FRAMES": "true",
                "FRAME_ANALYZER_DEBUG_FRAME_DIR": "custom/debug-frames",
            },
            clear=False,
        ):
            from frame_analyzer.legacy_agent import _resolve_debug_frame_dir

            self.assertEqual(_resolve_debug_frame_dir(), "custom/debug-frames")


if __name__ == "__main__":
    unittest.main()
