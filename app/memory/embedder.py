from typing import List, Optional

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logging import get_logger

log = get_logger("embedder")

class EmbedderSettings(BaseSettings):
    # Same env as ``Settings.ollama_base_url`` / `.env` ``OLLAMA_BASE_URL``
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class OllamaEmbedder:
    """Generates 768d embeddings using local Ollama nomic-embed-text model."""

    def __init__(self, settings: Optional[EmbedderSettings] = None, http_client: Optional[httpx.AsyncClient] = None):
        self._settings = settings or EmbedderSettings()
        # Create a new client if none provided, but it's recommended to inject one from the app lifespan
        self._http_client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def get_embedding(self, text: str) -> List[float]:
        """
        Takes an English text and returns a 768-dimensional float vector.
        """
        if not text or not text.strip():
            log.warning("Empty text provided to embedder.")
            return [0.0] * 768

        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/embeddings"
        payload = {
            "model": self._settings.ollama_embed_model,
            "prompt": text
        }

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            
            if not embedding or len(embedding) != 768:
                log.error("Invalid embedding length returned from Ollama: %s", len(embedding))
                return [0.0] * 768
                
            return embedding
        except Exception as exc:
            log.error("Failed to fetch embedding from Ollama: %s", exc)
            return [0.0] * 768

    async def close(self):
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
