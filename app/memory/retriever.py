import logging
from typing import List, Dict, Any, Optional
from app.memory.embedder import OllamaEmbedder
from app.memory.database import SupabaseDatabase

log = logging.getLogger(__name__)

class RAGRetriever:
    """Manages retrieving and storing memories in Supabase using pgvector."""

    def __init__(self, db: SupabaseDatabase, embedder: OllamaEmbedder):
        self.db = db
        self.embedder = embedder

    async def add_memory(self, ticker: str, memory_type: str, context: str, outcome: Optional[str] = None, pnl_percent: Optional[float] = None) -> bool:
        """Adds a new memory to trade_memories table."""
        pool = self.db.get_pool()
        if pool is None:
            log.warning("Database pool is not available. Memory not saved.")
            return False
            
        try:
            # Generate embedding
            embedding = await self.embedder.get_embedding(context)
            
            query = """
            INSERT INTO trade_memories 
            (ticker, memory_type, context, outcome, pnl_percent, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            # pgvector integration in asyncpg accepts native python lists
            async with pool.acquire() as conn:
                await conn.execute(query, ticker, memory_type, context, outcome, pnl_percent, embedding)
                
            log.info("Successfully added %s memory for %s", memory_type, ticker)
            return True
        except Exception as exc:
            log.error("Failed to add memory for %s: %s", ticker, exc)
            return False

    async def search_similar_memories(self, ticker: str, query_text: str, top_k: int = 5, match_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Searches for similar memories using pgvector cosine similarity."""
        pool = self.db.get_pool()
        if pool is None:
            log.warning("Database pool is not available. Cannot search memories.")
            return []
            
        try:
            # Generate embedding for query
            query_embedding = await self.embedder.get_embedding(query_text)
            
            # Call match_trade_memories function
            # Arguments: query_embedding, match_threshold, match_count, p_ticker
            query = """
            SELECT * FROM match_trade_memories($1, $2, $3, $4)
            """
            
            async with pool.acquire() as conn:
                records = await conn.fetch(query, query_embedding, match_threshold, top_k, ticker)
                
            results = []
            for record in records:
                results.append({
                    "id": record["id"],
                    "ticker": record["ticker"],
                    "memory_type": record["memory_type"],
                    "context": record["context"],
                    "outcome": record["outcome"],
                    "pnl_percent": record["pnl_percent"],
                    "similarity": record["similarity"]
                })
                
            return results
        except Exception as exc:
            log.error("Failed to search memories for %s: %s", ticker, exc)
            return []

    async def list_recent_memories(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """List most recent memories for ticker (deterministic; no vector search).

        This is used by the Telegram ``/memory SYMBOL`` command. Vector similarity is
        intentionally NOT used here because users expect "latest memories", not an
        embedding-dependent subset.
        """
        pool = self.db.get_pool()
        if pool is None:
            log.warning("Database pool is not available. Cannot list memories.")
            return []

        lim = max(1, min(int(limit), 50))
        try:
            query = """
            SELECT id, ticker, memory_type, context, outcome, pnl_percent, created_at
            FROM trade_memories
            WHERE ticker = $1
            ORDER BY created_at DESC
            LIMIT $2
            """
            async with pool.acquire() as conn:
                records = await conn.fetch(query, ticker, lim)

            results: list[dict] = []
            for record in records:
                results.append(
                    {
                        "id": record["id"],
                        "ticker": record["ticker"],
                        "memory_type": record["memory_type"],
                        "context": record["context"],
                        "outcome": record["outcome"],
                        "pnl_percent": record["pnl_percent"],
                        "created_at": record["created_at"].isoformat() if record["created_at"] else None,
                    }
                )
            return results
        except Exception as exc:
            log.error("Failed to list recent memories for %s: %s", ticker, exc)
            return []
