"""Agent-Rag: governed retrieval pipelines for AGenNext agents."""

from .contracts import RetrievalMode, RetrievalRequest, SourceDocument, TrustTier
from .pipeline import RagPipeline

__all__ = ["RagPipeline", "RetrievalMode", "RetrievalRequest", "SourceDocument", "TrustTier"]
__version__ = "0.1.0"
