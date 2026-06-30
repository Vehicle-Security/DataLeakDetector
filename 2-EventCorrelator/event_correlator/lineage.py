from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CorrelatorContext, FileLineage, NormalizedLogEvent
from .utils import extract_parent_candidates, get_path_basename, normalize_file_path


@dataclass
class LineageState:
    sensitive_roots: set[str]
    direct_mappings: dict[str, str] = field(default_factory=dict)
    root_mappings: dict[str, str] = field(default_factory=dict)
    artifact_instances: list[dict[str, str]] = field(default_factory=list)

    def add_mapping(
        self,
        parent_path: str,
        child_path: str,
        *,
        timestamp: str = "",
        event_type: str = "",
        source: str = "",
    ) -> None:
        parent = normalize_file_path(parent_path)
        child = normalize_file_path(child_path)
        if not parent or not child or parent == child:
            return

        self.direct_mappings[child] = parent
        self.root_mappings[child] = self.resolve_root(parent) or parent
        self.artifact_instances.append(
            {
                "artifact_id": self._artifact_id(child, timestamp),
                "path": child,
                "parent_path": parent,
                "root_path": self.root_mappings[child],
                "timestamp": timestamp,
                "event_type": event_type,
                "source": source or "lineage_mapping",
            }
        )

    @staticmethod
    def _artifact_id(path: str, timestamp: str) -> str:
        normalized = normalize_file_path(path).lower()
        evidence_time = str(timestamp or "").replace("T", " ").split(".")[0]
        return f"{normalized}@{evidence_time}" if evidence_time else normalized

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

    def known_artifacts(self) -> set[str]:
        artifacts = set(self.sensitive_roots)
        artifacts.update(self.direct_mappings.keys())
        artifacts.update(self.direct_mappings.values())
        artifacts.update(self.root_mappings.keys())
        artifacts.update(self.root_mappings.values())
        return {normalize_file_path(item) for item in artifacts if normalize_file_path(item)}

    def export(self) -> FileLineage:
        full_chains: dict[str, str] = {}
        for child_path in sorted(self.direct_mappings):
            full_chain = self.build_full_chain(child_path)
            if full_chain:
                full_chains[child_path] = full_chain

        return FileLineage(
            direct_file_mappings=dict(sorted(self.direct_mappings.items())),
            full_file_mapping_chains=full_chains,
            artifact_instances=list(self.artifact_instances),
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
                state.add_mapping(
                    parent_path,
                    file_path,
                    timestamp=log_event.timestamp_text,
                    event_type=event_type,
                    source="explicit_parent_field",
                )
                return

        related_paths = log_event.raw.get("related_paths", [])
        if isinstance(related_paths, list):
            for parent_path in related_paths:
                normalized_parent = normalize_file_path(str(parent_path or ""))
                if normalized_parent and normalized_parent != file_path:
                    state.add_mapping(
                        normalized_parent,
                        file_path,
                        timestamp=log_event.timestamp_text,
                        event_type=event_type,
                        source="related_paths",
                    )
                    return

        if event_type in {"created", "renamed", "copied", "converted", "compressed"}:
            inferred_parent = self._infer_parent_from_known_artifacts(file_path, state.known_artifacts())
            if inferred_parent:
                state.add_mapping(
                    inferred_parent,
                    file_path,
                    timestamp=log_event.timestamp_text,
                    event_type=event_type,
                    source="known_artifact_stem_inference",
                )

    def _infer_parent_from_known_artifacts(self, file_path: str, known_artifacts: set[str]) -> str:
        normalized_path = normalize_file_path(file_path)
        target_base = normalized_path.rsplit("/", 1)[-1].lower()
        target_no_ext = target_base.rsplit(".", 1)[0]
        target_dir = normalized_path.rsplit("/", 1)[0].lower() if "/" in normalized_path else ""

        best_match = ""
        best_score = 0
        for artifact_path in known_artifacts:
            normalized_artifact = normalize_file_path(artifact_path)
            if not normalized_artifact or normalized_artifact == normalized_path:
                continue
            artifact_base = normalized_artifact.rsplit("/", 1)[-1].lower()
            artifact_no_ext = artifact_base.rsplit(".", 1)[0]
            artifact_dir = normalized_artifact.rsplit("/", 1)[0].lower() if "/" in normalized_artifact else ""

            score = 0
            if artifact_no_ext and target_no_ext.startswith(artifact_no_ext):
                score += 4 + min(len(artifact_no_ext), 30)
            elif target_no_ext and artifact_no_ext.startswith(target_no_ext):
                score += 2 + min(len(target_no_ext), 20)
            if artifact_dir and target_dir and artifact_dir == target_dir:
                score += 3
            if artifact_no_ext and artifact_no_ext in target_no_ext:
                score += 2
            if score > best_score:
                best_match = normalized_artifact
                best_score = score

        return best_match if best_score >= 6 else ""
