import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "02-EventCorrelator"

if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


from event_correlator.lineage import LineageState  # noqa: E402


class LineageMappingLinkTests(unittest.TestCase):
    def test_build_full_mapping_chain_supports_multiple_files(self):
        lineage = LineageState(
            sensitive_roots={"C:/Users/test/Desktop/orig.xlsx"},
            direct_mappings={
                "C:/Users/test/Desktop/part1.xlsx": "C:/Users/test/Desktop/orig.xlsx",
                "C:/Users/test/Desktop/part2.xlsx": "C:/Users/test/Desktop/orig.xlsx",
            },
            root_mappings={
                "C:/Users/test/Desktop/part1.xlsx": "C:/Users/test/Desktop/orig.xlsx",
                "C:/Users/test/Desktop/part2.xlsx": "C:/Users/test/Desktop/orig.xlsx",
            },
        )

        mapping_links = [
            lineage.build_full_chain("C:/Users/test/Desktop/part1.xlsx"),
            lineage.build_full_chain("C:/Users/test/Desktop/part2.xlsx"),
        ]

        self.assertEqual(
            mapping_links,
            [
                "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/part1.xlsx",
                "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/part2.xlsx",
            ],
        )


if __name__ == "__main__":
    unittest.main()
