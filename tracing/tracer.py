import threading
from datetime import datetime


class Tracer:
    def __init__(self):
        self._lock   = threading.Lock()
        self._traces = {}    # query_id → [step, ...]
        self._global = {"agent_calls": {}, "latencies": {}, "total_calls": 0}

    def add(self, query_id: str, agent: str, message: str, status: str, latency_ms: int):
        step = {
            "agent":      agent,
            "message":    message,
            "status":     status,
            "latency_ms": latency_ms,
            "ts":         datetime.utcnow().strftime("%H:%M:%S"),
        }
        with self._lock:
            self._traces.setdefault(query_id, []).append(step)
            self._global["agent_calls"].setdefault(agent, 0)
            self._global["agent_calls"][agent] += 1
            self._global["latencies"].setdefault(agent, [])
            if latency_ms > 0:
                self._global["latencies"][agent].append(latency_ms)
            self._global["total_calls"] += 1

    def get(self, query_id: str) -> list:
        with self._lock:
            return list(self._traces.get(query_id, []))

    def stats(self) -> dict:
        with self._lock:
            avg_lat = {
                agent: round(sum(v) / len(v)) if v else 0
                for agent, v in self._global["latencies"].items()
            }
            return {
                "agent_calls":    dict(self._global["agent_calls"]),
                "avg_latency_ms": avg_lat,
                "total_calls":    self._global["total_calls"],
                "total_queries":  len(self._traces),
            }