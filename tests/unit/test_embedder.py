import pytest
import os
from app.memory.embedder import OllamaEmbedder

@pytest.mark.asyncio
async def test_embedder_returns_768d_vector():
    embedder = OllamaEmbedder()
    try:
        # Test valid string
        vector = await embedder.get_embedding("This is a test document for embedding.")
        assert isinstance(vector, list)
        assert len(vector) == 768
        assert all(isinstance(x, float) for x in vector)
        
        # Test empty string fallback
        empty_vector = await embedder.get_embedding("")
        assert len(empty_vector) == 768
        assert empty_vector == [0.0] * 768
    finally:
        await embedder.close()
