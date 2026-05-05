"""Read-only inspection helpers for stored trajectories and snapshots."""

from __future__ import annotations

from trajpatch.storage.models import EpisodicMemorySnapshot
from trajpatch.storage.repository import TrajPatchStore


class Inspector:
    def __init__(self, store: TrajPatchStore) -> None:
        self.store = store

    def trajectory(self, trajectory_id: str) -> dict:
        return self.store.inspect_trajectory(trajectory_id)

    def snapshot(self, snapshot_id: str) -> dict:
        snapshot = self.store.session.get(EpisodicMemorySnapshot, snapshot_id)
        if snapshot is None:
            return {}
        claims = self.store.list_claims_for_snapshot(snapshot_id)
        ops = self.store.list_claim_ops_for_snapshot(snapshot_id)
        payload = snapshot.__dict__.copy()
        payload.pop("_sa_instance_state", None)
        payload["memory_type"] = "episodic"
        payload["claims"] = [claim.__dict__.copy() for claim in claims]
        payload["ops"] = [op.__dict__.copy() for op in ops]
        for row in payload["claims"] + payload["ops"]:
            row.pop("_sa_instance_state", None)
        return payload
