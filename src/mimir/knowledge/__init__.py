"""Knowledge and memory subsystem (ADR 11)."""

from mimir.knowledge.embeddings import Embedder, HashingEmbedder, get_embedder
from mimir.knowledge.importers import ConversationImporter
from mimir.knowledge.index import KnowledgeIndex, MemoryFilters, get_knowledge_index
from mimir.knowledge.promotion import MemoryPromoter, PromotionRefused
from mimir.knowledge.retrieval import (
    MemoryConflict,
    MemoryRetriever,
    RetrievalResult,
    build_filters,
)
from mimir.knowledge.store import (
    DocumentMetadata,
    KnowledgeStore,
    MemoryDocument,
    MemoryLayer,
    get_knowledge_store,
)

__all__ = [
    "ConversationImporter",
    "DocumentMetadata",
    "Embedder",
    "HashingEmbedder",
    "KnowledgeIndex",
    "KnowledgeStore",
    "MemoryConflict",
    "MemoryDocument",
    "MemoryFilters",
    "MemoryLayer",
    "MemoryPromoter",
    "MemoryRetriever",
    "PromotionRefused",
    "RetrievalResult",
    "build_filters",
    "get_embedder",
    "get_knowledge_index",
    "get_knowledge_store",
]
