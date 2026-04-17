from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CorrelatorContext, FileLineage, NormalizedLogEvent
from .utils import extract_parent_candidates, get_path_basename, normalize_file_path


@dataclass
class LineageState:
    sensitive_roots: set[str]
    direct_mappings: dict[str, str] = field(default_factory=dict)
    root_mappings: dict[str, str] = field(default_factory=dict)

    def add_mapping(self, parent_path: str, child_path: str) -> None:
        parent = normalize_file_path(parent_path)
        child = normalize_file_path(child_path)
        if not parent or not child or parent == child:
            return

        self.direct_mappings[child] = parent
        self.root_mappings[child] = self.resolve_root(parent) or parent

    def resolve_root(self, file_path: str, max_depth: int = 10) -> str:
        current = normalize_file_path(file_path)
        if not current:
            return ""

        if current in self.sensitive_roots:
            return current

        if current in self.root_mappings:
            return self.root_mappings[current]

        seen = set()
        for _ in range(max_depth):
            if not current or current in seen:
                return ""
            seen.add(current)
            parent = self.direct_mappings.get(current, "")
            if not parent:
                return current if current in self.sensitive_roots else ""
            if parent in self.sensitive_roots:
                return parent
            current = parent
        return ""

    def build_full_chain(self, file_path: str, max_depth: int = 10) -> str:
        current = normalize_file_path(file_path)
        if not current:
            return ""

        if current not in self.direct_mappings and current not in self.sensitive_roots:
            return ""

        chain = [current]
        seen = {current}
        for _ in range(max_depth):
            parent = self.direct_mappings.get(current, "")
            if not parent or parent in seen:
                break
            chain.insert(0, parent)
            seen.add(parent)
            current = parent
        return " -> ".join(chain)

    def resolve_by_basename(self, file_name: str) -> str:
        target_name = get_path_basename(file_name).lower()
        if not target_name:
            return ""

        candidates = list(self.direct_mappings.keys()) + list(self.sensitive_roots)
        for candidate in candidates:
            if get_path_basename(candidate).lower() == target_name:
                return candidate
        return ""

    def export(self) -> FileLineage:
        full_chains: dict[str, str] = {}
        for child_path in sorted(self.direct_mappings):
            full_chain = self.build_full_chain(child_path)
            if full_chain:
                full_chains[child_path] = full_chain

        return FileLineage(
            direct_file_mappings=dict(sorted(self.direct_mappings.items())),
            full_file_mapping_chains=full_chains,
        )


class LineageBuilder:
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    def build(self, context: CorrelatorContext) -> LineageState:
        state = LineageState(sensitive_roots=set(context.sensitive_files))

        for log_event in context.normalized_logs:
            self._update_from_log_event(state, log_event)

        return state

    def _update_from_log_event(self, state: LineageState, log_event: NormalizedLogEvent) -> None:
        file_path = normalize_file_path(log_event.file_path)
        if not file_path:
            return

        event_type = str(log_event.event_type or "").lower()

        parent_candidates = extract_parent_candidates(log_event.raw)
        for parent_path in parent_candidates:
            if parent_path != file_path:
                state.add_mapping(parent_path, file_path)
                return

        related_paths = log_event.raw.get("related_paths", [])
        if isinstance(related_paths, list):
            for parent_path in related_paths:
                normalized_parent = normalize_file_path(str(parent_path or ""))
                if normalized_parent and normalized_parent != file_path:
                    state.add_mapping(normalized_parent, file_path)
                    return

        if event_type in {"created", "renamed", "copied", "converted", "compressed"}:
            inferred_parent = self._infer_parent_from_sensitive_roots(file_path, state.sensitive_roots)
            if inferred_parent:
                state.add_mapping(inferred_parent, file_path)

    def _infer_parent_from_sensitive_roots(self, file_path: str, sensitive_roots: set[str]) -> str:
        normalized_path = normalize_file_path(file_path)
        target_base = normalized_path.rsplit("/", 1)[-1].lower()
        target_no_ext = target_base.rsplit(".", 1)[0]

        best_match = ""
        best_length = -1
        for root_path in sensitive_roots:
            normalized_root = normalize_file_path(root_path)
            root_base = normalized_root.rsplit("/", 1)[-1].lower()
            root_no_ext = root_base.rsplit(".", 1)[0]
            if root_no_ext and target_no_ext.startswith(root_no_ext) and len(root_no_ext) > best_length:
                best_match = normalized_root
                best_length = len(root_no_ext)

        return best_match
