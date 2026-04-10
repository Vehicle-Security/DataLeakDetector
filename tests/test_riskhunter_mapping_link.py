import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeManager:
    def __init__(self, mappings):
        self.mappings = mappings

    def get_mapping_chain(self, file_path: str):
        return self.mappings.get(file_path)


class RiskHunterMappingLinkTests(unittest.TestCase):
    def test_build_upload_content_mapping_link_supports_multiple_files(self):
        module_dir = REPO_ROOT / "3-RiskHunter"
        sys.path.insert(0, str(module_dir))
        sys.path.insert(0, str(REPO_ROOT / "2-FileTracker"))
        try:
            nodes_module = load_module(
                "upload_detector_nodes_mapping_link_test",
                module_dir / "upload_detector_nodes.py",
            )
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        manager = FakeManager(
            {
                "C:/Users/test/Desktop/part1.xlsx": "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/part1.xlsx",
                "C:/Users/test/Desktop/part2.xlsx": "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/part2.xlsx",
            }
        )

        mapping_link = nodes_module._build_upload_content_mapping_link(
            manager=manager,
            upload_content="part1.xlsx, part2.xlsx",
            current_event={"file_path": "C:/Users/test/Desktop/part1.xlsx"},
            event_data={"time_range": "2026-03-27 12:31:48 - 2026-03-27 12:32:17"},
            log_events=[],
        )

        self.assertEqual(
            mapping_link,
            "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/part1.xlsx | "
            "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/part2.xlsx",
        )


if __name__ == "__main__":
    unittest.main()
