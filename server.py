"""
BGE-M3 Embedding API Server
===========================
Serves BAAI/bge-m3 dense embeddings via FastAPI.

Runs as a separate process from the TTS server to avoid interfering with it:
  - Separate CUDA context (own process)
  - Capped GPU memory (MAX_MEMORY_FRACTION env var, default 0.40)
  - Same expandable_segments allocator setting as TTS

Endpoints:
  POST /embed         - Batch embed a list of texts
  POST /embed/query   - Embed a single query (convenience)
  GET  /health        - Health + VRAM usage

Install:
  uv venv venv && source venv/bin/activate
  uv pip install -r requirements.txt

Run:
  python server.py --host 0.0.0.0 --port 8060
"""

import argparse
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

# Match the TTS server's allocator config — reduces fragmentation when
# two CUDA processes share the same GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bge-m3-server")

# ---------------------------------------------------------------------------
# Config (env vars)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-m3")
API_KEY = os.environ.get("API_KEY", "")

# Hard cap on this process's GPU memory (fraction of total VRAM).
# Keeps the embedding model from starving the TTS model on a shared GPU.
# 0.15 of 40 GB (A100) = ~6 GB — enough for BGE-M3 fp16 + a large batch.
MAX_MEMORY_FRACTION = float(os.environ.get("MAX_MEMORY_FRACTION", "0.15"))

# Default and maximum batch sizes used inside model.encode().
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "128"))

# Token truncation. 1024 is plenty for typical RAG chunks; bump to 8192 only
# if you really embed very long documents (much slower).
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))

# Max concurrent requests in the pipeline (1 running + queued).
PIPELINE_DEPTH = int(os.environ.get("PIPELINE_DEPTH", "4"))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
_lock = asyncio.Lock()
_sema = asyncio.Semaphore(PIPELINE_DEPTH)


def _verify_api_key(api_key: Optional[str]) -> None:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of texts to embed. Send in batches for throughput.",
        json_schema_extra={"examples": [["first chunk of text", "second chunk of text"]]},
    )
    normalize: bool = Field(
        default=True,
        description="L2-normalize output vectors (recommended for cosine similarity search).",
    )
    max_length: int = Field(
        default=DEFAULT_MAX_LENGTH, ge=16, le=8192,
        description="Truncate inputs to this many tokens. Shorter = faster.",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE,
        description="Internal batch size for the model forward pass.",
    )


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    count: int
    processing_time: float


class QueryRequest(BaseModel):
    text: str = Field(..., description="Single query text to embed.")
    normalize: bool = True
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=8192)


class QueryResponse(BaseModel):
    embedding: List[float]
    dimension: int
    processing_time: float


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _encode_sync(
    texts: List[str],
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    with torch.inference_mode():
        out = model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    vecs = out["dense_vecs"]
    # BGE-M3 returns L2-normalized dense vectors by default. Re-normalize
    # only if the caller explicitly wants it (cheap safety net).
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs = vecs / norms
    return vecs


async def _run_encode(texts, batch_size, max_length, normalize) -> np.ndarray:
    try:
        await asyncio.wait_for(_sema.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, try again")
    try:
        async with _lock:
            try:
                return await asyncio.to_thread(
                    _encode_sync, texts, batch_size, max_length, normalize
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    logger.warning("CUDA OOM detected, clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    return await asyncio.to_thread(
                        _encode_sync, texts, batch_size, max_length, normalize
                    )
                raise e
    finally:
        _sema.release()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from FlagEmbedding import BGEM3FlagModel

    if torch.cuda.is_available() and 0.0 < MAX_MEMORY_FRACTION < 1.0:
        torch.cuda.set_per_process_memory_fraction(MAX_MEMORY_FRACTION, 0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(
            "GPU memory capped at %.0f%% of %.1f GB (~%.1f GB max)",
            MAX_MEMORY_FRACTION * 100, total_gb, total_gb * MAX_MEMORY_FRACTION,
        )

    use_fp16 = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s (fp16=%s) ...", MODEL_NAME, device, use_fp16)
    t0 = time.time()
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=use_fp16)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Warmup — triggers lazy weights init and caches CUDA kernels for
    # the common small-batch path. Doing two shapes so both are cached.
    logger.info("Running warmup...")
    t_w = time.time()
    with torch.no_grad():
        _ = model.encode(["warmup"], batch_size=1, max_length=128, return_dense=True)
        _ = model.encode(
            ["warmup " * 64] * 8, batch_size=8, max_length=512, return_dense=True
        )
    logger.info("Warmup done in %.1fs — server ready", time.time() - t_w)

    yield


app = FastAPI(
    title="BGE-M3 Embedding Server",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    device_name = "cpu"
    vram_mb = 0
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_mb = round(torch.cuda.memory_allocated(0) / 1e6, 1)
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": device_name,
        "vram_mb": vram_mb,
        "memory_fraction_cap": MAX_MEMORY_FRACTION,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, x_api_key: Optional[str] = Header(None)):
    _verify_api_key(x_api_key)
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")

    t0 = time.time()
    vecs = await _run_encode(req.texts, req.batch_size, req.max_length, req.normalize)
    elapsed = time.time() - t0

    embeddings = vecs.tolist()
    dim = len(embeddings[0]) if embeddings else 0
    logger.info(
        "embed: %d texts, dim=%d, %.3fs (%.1f texts/s)",
        len(req.texts), dim, elapsed, len(req.texts) / elapsed if elapsed > 0 else 0,
    )

    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        dimension=dim,
        count=len(embeddings),
        processing_time=round(elapsed, 4),
    )


@app.post("/embed/query", response_model=QueryResponse)
async def embed_query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    _verify_api_key(x_api_key)
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    t0 = time.time()
    vecs = await _run_encode([req.text], 1, req.max_length, req.normalize)
    elapsed = time.time() - t0

    embedding = vecs[0].tolist()
    logger.info("embed/query: dim=%d, %.3fs", len(embedding), elapsed)

    return QueryResponse(
        embedding=embedding,
        dimension=len(embedding),
        processing_time=round(elapsed, 4),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-M3 Embedding Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8060)))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--api-key", default=None, help="Overrides API_KEY env var")
    args = parser.parse_args()

    if args.api_key:
        API_KEY = args.api_key
    MODEL_NAME = args.model

    uvicorn.run("server:app", host=args.host, port=args.port, workers=1, log_level="info")
