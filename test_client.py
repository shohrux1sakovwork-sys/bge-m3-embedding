"""
Simple test/usage client for the BGE-M3 embedding server.

Run the server first:
    python server.py --port 8060

Then:
    python test_client.py                 # runs all tests
    python test_client.py --url http://your-server:8060
"""

import argparse
import time

import numpy as np
import requests


def embed_batch(url: str, texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Embed a list of texts. Use this for indexing document chunks."""
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.post(
        f"{url}/embed",
        json={"texts": texts, "normalize": True, "max_length": 1024, "batch_size": 32},
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  -> {data['count']} vectors, dim={data['dimension']}, {data['processing_time']}s")
    return data["embeddings"]


def embed_query(url: str, text: str, api_key: str | None = None) -> list[float]:
    """Embed a single query text. Use this at search time."""
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.post(
        f"{url}/embed/query",
        json={"text": text, "normalize": True},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"  -> dim={data['dimension']}, {data['processing_time']}s")
    return data["embedding"]


def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_tests(url: str, api_key: str | None):
    print(f"== BGE-M3 client against {url} ==\n")

    # 1. Health
    print("[1] GET /health")
    resp = requests.get(f"{url}/health", timeout=10)
    print(" ", resp.json(), "\n")

    # 2. Batch embed (the RAG indexing path — embed many chunks at once)
    print("[2] POST /embed — batch indexing")
    chunks = [
        "The RTX 5070 is a graphics card by NVIDIA based on the Blackwell architecture.",
        "BGE-M3 is a multilingual embedding model that supports 100+ languages.",
        "Qovunlar yozda pishadi va juda shirin bo'ladi.",  # Uzbek
        "Машинное обучение активно развивается в последние годы.",  # Russian
        "FastAPI is a modern Python web framework for building APIs.",
    ]
    t0 = time.time()
    vecs = embed_batch(url, chunks, api_key)
    print(f"  batch wall time: {time.time() - t0:.3f}s\n")

    # 3. Query embed (the RAG search path)
    print("[3] POST /embed/query — search query")
    query = "Which GPU is based on Blackwell?"
    q_vec = embed_query(url, query, api_key)

    # 4. Cosine similarity — sanity check that the right chunk scores highest
    print("\n[4] Cosine similarity ranking")
    scores = [(cosine(q_vec, v), chunks[i]) for i, v in enumerate(vecs)]
    scores.sort(reverse=True)
    for s, text in scores:
        print(f"  {s:+.4f}  {text[:70]}")

    # 5. Large batch stress test — simulates indexing a doc
    print("\n[5] POST /embed — stress test, 200 texts")
    big = [f"Test chunk number {i} with some arbitrary filler content." for i in range(200)]
    t0 = time.time()
    vecs = embed_batch(url, big, api_key)
    dt = time.time() - t0
    print(f"  200 texts in {dt:.2f}s = {200 / dt:.1f} texts/s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8060")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    run_tests(args.url.rstrip("/"), args.api_key)
