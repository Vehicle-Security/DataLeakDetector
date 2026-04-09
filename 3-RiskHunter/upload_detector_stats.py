from typing import MutableMapping, Any


def sync_processed_statistics(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    statistics = state.setdefault("statistics", {})
    statistics["total_events_processed"] = state.get("processed_count", 0)
    return state
