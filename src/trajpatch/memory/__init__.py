"""Memory extraction, parsing, retrieval, and rendering."""

__all__ = ["MemoryOrchestrator", "RetrievalEngine"]


def __getattr__(name: str):
    if name == "MemoryOrchestrator":
        from .orchestrator import MemoryOrchestrator

        return MemoryOrchestrator
    if name == "RetrievalEngine":
        from .retrieval import RetrievalEngine

        return RetrievalEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
