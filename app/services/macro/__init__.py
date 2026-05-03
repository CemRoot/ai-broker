"""Macro / sentiment data sources (Faz 3 — extended data layer).

Currently exposes the **CNN Fear & Greed** client which alone covers:

- composite Fear/Greed score (0–100) + rating bucket
- put/call options ratio sub-component
- VIX sub-component (cross-check for our own yfinance call)
- market momentum, breadth, junk-bond, safe-haven sub-components

That single endpoint replaces three separate integrations from the canonical
plan (`AI_BROKER_PROJECT.md` "Korku & Açgözlülük göstergeleri") and is free,
unauthenticated, and JSON.
"""

from .cnn_fear_greed import CNNFearGreedClient, CNNFearGreedSnapshot  # noqa: F401
