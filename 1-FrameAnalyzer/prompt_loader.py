import importlib.util
from pathlib import Path


def _load_local_prompts():
    prompts_path = Path(__file__).with_name("prompts.py")
    spec = importlib.util.spec_from_file_location("frame_analyzer_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load prompts from {prompts_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPTS = _load_local_prompts()
