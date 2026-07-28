"""Pluggable embedding backends for memory retrieval (ADR 11.4, 18.4).

``mimir.llm`` does not exist yet and the whole point of ADR 18 is that MIMIR
runs on whatever local runtime is present, so this module keeps the contract
tiny: an :class:`Embedder` is anything with ``dimensions``, ``name``, and an
``embed(texts) -> list[list[float]]`` method.

Two implementations ship here:

* :class:`OpenAICompatEmbedder` talks to any ``/v1/embeddings`` endpoint, which
  covers Ollama, llama.cpp, LM Studio, LiteLLM, and vLLM.
* :class:`HashingEmbedder` is a deterministic hashed bag-of-ngrams projection.
  It needs no model, no network, and no download, so semantic retrieval still
  degrades to something useful rather than to nothing when no runtime is up.

The retrieval path never requires an embedder: keyword search alone is a valid
mode, and :func:`get_embedder` may return ``None``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)

EMBEDDING_DTYPE = np.float32


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding contract."""

    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-norm vector per input text."""
        ...


def to_blob(vector: list[float] | np.ndarray) -> bytes:
    return np.asarray(vector, dtype=EMBEDDING_DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a stacked matrix."""
    if matrix.size == 0:
        return np.zeros(0, dtype=EMBEDDING_DTYPE)
    q = np.asarray(query, dtype=EMBEDDING_DTYPE).reshape(-1)
    q_norm = np.linalg.norm(q) or 1.0
    return (matrix @ q) / (np.linalg.norm(matrix, axis=1).clip(min=1e-9) * q_norm)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or",
        "that", "the", "their", "then", "there", "these", "this", "to", "was",
        "were", "will", "with", "you", "your",
    ]
)


def tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic hashed projection. No model required.

    Unigrams plus adjacent bigrams are hashed into a fixed number of buckets
    with a signed hash, sublinear term frequency, and L2 normalisation. This is
    a lexical-overlap embedding rather than a semantic one, so it will not match
    paraphrases; it exists so hybrid retrieval has a working second signal on a
    machine with no embedding model installed.
    """

    dimensions: int = 768
    name: str = "hashing-fallback"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dimensions, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = np.zeros((len(texts), self.dimensions), dtype=EMBEDDING_DTYPE)
        for row, text in enumerate(texts):
            tokens = tokenise(text)
            if not tokens:
                continue
            grams = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
            counts: dict[str, int] = {}
            for gram in grams:
                counts[gram] = counts.get(gram, 0) + 1
            for gram, count in counts.items():
                index, sign = self._bucket(gram)
                out[row, index] += sign * (1.0 + np.log(count))
        return normalise(out).tolist()


@dataclass(slots=True)
class OpenAICompatEmbedder:
    """Client for any OpenAI-compatible ``/v1/embeddings`` endpoint."""

    base_url: str
    model: str
    dimensions: int = 768
    api_key: str | None = None
    timeout_s: float = 60.0
    batch_size: int = 32
    name: str = ""

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.name:
            self.name = f"{self.model}@{self.base_url}"

    def _endpoint(self) -> str:
        suffix = "/embeddings" if self.base_url.endswith("/v1") else "/v1/embeddings"
        return self.base_url + suffix

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout_s) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(
                    self._endpoint(),
                    json={"model": self.model, "input": batch},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                rows = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
                vectors.extend([list(row["embedding"]) for row in rows])
        if vectors:
            matrix = normalise(np.asarray(vectors, dtype=EMBEDDING_DTYPE))
            # Trust the server over the configured dimension count.
            self.dimensions = int(matrix.shape[1])
            return matrix.tolist()
        return vectors

    def probe(self) -> bool:
        """Cheap liveness check so the index can fall back without a stack trace."""
        try:
            self.embed(["ping"])
        except Exception as exc:  # noqa: BLE001 - any failure means "not available"
            log.info("embedder_unavailable", model=self.model, error=str(exc))
            return False
        return True


@dataclass(slots=True)
class NullEmbedder:
    """Placeholder used when embeddings are disabled."""

    dimensions: int = 1
    name: str = "null"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def get_embedder(
    settings: Settings | None = None,
    *,
    override: Embedder | None = None,
    allow_remote: bool = True,
    probe: bool = True,
) -> Embedder | None:
    """Resolve the configured embedder, falling back to hashing.

    Returns ``None`` when ``knowledge.embeddings_enabled`` is false, which the
    index and retriever treat as keyword-only mode.
    """
    if override is not None:
        return override
    cfg = settings or get_settings()
    if not cfg.knowledge.embeddings_enabled:
        return None

    dimensions = cfg.knowledge.embedding_dimensions
    if allow_remote:
        alias = cfg.models.routing.embedding
        profile = cfg.models.profiles.get(alias)
        if profile is not None:
            remote = OpenAICompatEmbedder(
                base_url=profile.base_url,
                model=profile.model,
                api_key=profile.api_key,
                dimensions=dimensions,
                timeout_s=profile.request_timeout_s,
            )
            if not probe or remote.probe():
                return remote
    return HashingEmbedder(dimensions=dimensions)
