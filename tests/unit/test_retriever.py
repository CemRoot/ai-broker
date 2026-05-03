import pytest
import math
from app.memory.embedder import OllamaEmbedder
from app.memory.database import SupabaseDatabase, DatabaseSettings
from app.memory.retriever import RAGRetriever

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

@pytest.mark.asyncio
async def test_cosine_similarity_logic():
    """Test raw cosine similarity output of our embedder to verify A-B > 0.7, A-C < 0.5."""
    embedder = OllamaEmbedder()
    try:
        text_a = "The tech market rallied today because of strong earnings from NVDA."
        text_b = "Nvidia's excellent quarterly report caused technology stocks to surge."
        text_c = "Making a perfect chocolate cake requires exactly 3 eggs and good cocoa powder."
        
        vec_a = await embedder.get_embedding(text_a)
        vec_b = await embedder.get_embedding(text_b)
        vec_c = await embedder.get_embedding(text_c)
        
        sim_ab = cosine_similarity(vec_a, vec_b)
        sim_ac = cosine_similarity(vec_a, vec_c)
        
        # Similar sentences should have high cosine similarity (> 0.7)
        assert sim_ab > 0.7
        # Unrelated sentences should have lower cosine similarity (< 0.5)
        assert sim_ac < 0.5
    finally:
        await embedder.close()

@pytest.mark.asyncio
async def test_retriever_db_integration():
    """Integration test that runs if DB is available."""
    settings = DatabaseSettings()
    db_url = settings.supabase_db_url
    if not db_url:
        pytest.skip("SUPABASE_DB_URL not set, skipping DB integration test.")
        
    db = SupabaseDatabase(settings)
    await db.connect()
    
    embedder = OllamaEmbedder()
    retriever = RAGRetriever(db=db, embedder=embedder)
    
    try:
        # Add a fake memory
        success = await retriever.add_memory(
            ticker="TEST",
            memory_type="LESSON",
            context="This is a highly specific fake memory for unit testing purposes.",
            outcome="OPEN"
        )
        assert success is True
        
        # Search and verify it returns
        results = await retriever.search_similar_memories(
            ticker="TEST",
            query_text="highly specific fake memory test",
            top_k=1,
            match_threshold=0.5
        )
        
        assert len(results) > 0
        assert "fake memory for unit testing" in results[0]["context"]
        assert results[0]["similarity"] > 0.6
        
    finally:
        # Cleanup
        pool = db.get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM trade_memories WHERE ticker = 'TEST'")
        await db.close()
        await embedder.close()
