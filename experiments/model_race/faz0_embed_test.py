"""
Faz 0 — nomic-embed-text smoke test (Ollama local).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm
from ollama import Client as OllamaClient

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TEXT_A = "AMD stock RSI at 80, overbought signal, consider selling"
TEXT_B = "AMD price at peak, strong sell signal, RSI overbought"
TEXT_C = "Bitcoin surged 10% today, crypto market rally"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot(a, b) / (na * nb))


def main() -> None:
    host = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
    model = "nomic-embed-text"
    client = OllamaClient(host=host)

    texts = [TEXT_A, TEXT_B, TEXT_C]
    t0 = time.perf_counter()
    vectors: list[list[float]] = []
    for t in texts:
        resp = client.embed(model=model, input=t)
        vectors.append(list(resp.embeddings[0]))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    va, vb, vc = vectors
    sim_ab = cosine_similarity(va, vb)
    sim_ac = cosine_similarity(va, vc)
    dim = len(va)

    ok = sim_ab > 0.7 and sim_ac < 0.5
    verdict = "Embedding ÇALIŞIYOR ✅" if ok else "Embedding kalitesi düşük ⚠️"

    print(f"Model: {model}  |  Boyut: {dim}  |  3 embed süresi: {elapsed_ms:.1f} ms")
    print(f"A-B benzerlik: {sim_ab:.2f} (beklenen: yüksek, eşik > 0.7)")
    print(f"A-C benzerlik: {sim_ac:.2f} (beklenen: düşük, eşik < 0.5)")
    print(verdict)
    if not ok and sim_ab > sim_ac:
        print(f"(Teşhis: A-B > A-C farkı {sim_ab - sim_ac:.2f} — konu ayrımı tutarlı; eşikleri veya modeli ince ayar gerekebilir.)")


if __name__ == "__main__":
    main()
