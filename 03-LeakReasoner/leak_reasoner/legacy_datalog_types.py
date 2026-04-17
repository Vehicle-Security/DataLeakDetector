from dataclasses import dataclass


@dataclass
class LeakPath:
    start_op: str
    end_op: str
    leaking_proc: str
    leaked_file: str
    full_path: str
    leak_channel: str
    leak_timestamp: int

    def to_dict(self) -> dict:
        return {
            "start_op": self.start_op,
            "end_op": self.end_op,
            "leaking_proc": self.leaking_proc,
            "leaked_file": self.leaked_file,
            "full_path": self.full_path,
            "leak_channel": self.leak_channel,
            "leak_timestamp": self.leak_timestamp,
            "path_steps": self.full_path.split(" -> "),
        }
