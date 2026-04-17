import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parent

if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from main_v2 import main as pipeline_main


if __name__ == "__main__":
    pipeline_main()
