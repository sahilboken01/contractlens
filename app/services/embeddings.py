from functools import lru_cache

from app.core.config import get_settings


class EmbeddingConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise EmbeddingConfigurationError("Install the requirements to use the embedding model.") from exc
    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise EmbeddingConfigurationError(
            f"Unable to load embedding model '{model_name}'. Check the model name and model download access."
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _load_model(get_settings().embedding_model)
    try:
        vectors = [vector.tolist() for vector in model.embed(texts)]
    except Exception as exc:
        raise EmbeddingConfigurationError("Embedding generation failed.") from exc
    expected = get_settings().embedding_dimensions
    if any(len(vector) != expected for vector in vectors):
        actual = len(vectors[0]) if vectors else 0
        raise EmbeddingConfigurationError(
            f"Model returned {actual} dimensions but EMBEDDING_DIMENSIONS is set to {expected}."
        )
    return vectors
