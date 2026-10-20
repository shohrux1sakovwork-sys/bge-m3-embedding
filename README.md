# BGE-M3 Embedding Server

A self-contained FastAPI server for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) dense embeddings, designed to run **alongside the OmniVoice TTS server on the same GPU** without interfering with it.

Use it for RAG pipelines: embed chunks at indexing time, embed queries at search time, and store the vectors in any vector DB (Qdrant, Milvus, pgvector, Chroma, etc.).

## Why a separate server?

Running the embedding model in-process with TTS is fragile — CUDA OOM from a big batch, `torch.compile` cache conflicts, and blocking the event loop are all real risks. This server sidesteps that:

| Mechanism | What it does |
|---|---|
| **Separate process / own venv** | Separate CUDA context; a crash or OOM here cannot take down TTS |
| **`torch.cuda.set_per_process_memory_fraction(0.40)`** | Hard cap of ~40% of VRAM (~4.8 GB on a 12 GB RTX 5070). TTS is guaranteed the rest |
| **FP16 weights** | BGE-M3 is ~1.1 GB in fp16 vs ~2.3 GB in fp32 |
| **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | Same allocator mode as TTS — reduces fragmentation when two processes share the GPU |
| **`asyncio.Lock` + `Semaphore(4)`** | Serialises GPU work and fails fast with HTTP 503 when overloaded |
| **Different port (8060)** | TTS stays on 8050, embeddings on 8060 |

## Quick start

### Docker (recommended — matches the TTS setup)

```bash
cd bge_m3_server
cp .env.example .env          # optional, edit values
docker compose up -d --build
docker compose logs -f        # wait for "server ready"
```

The container shares `~/.cache/huggingface` with the host (and with the TTS container), so the model weights are downloaded once and reused.

Runs on **host networking**, same as TTS, so the service is on `http://localhost:8060` directly.

### Bare metal

```bash
cd bge_m3_server
./setup.sh                    # creates venv, installs deps
source venv/bin/activate
python server.py --port 8060  # starts server, warmup takes ~10s
```

First request is ready once you see `server ready` in the logs.

### Smoke test

```bash
# In another terminal
python test_client.py
```

Should print a health check, ranked cosine scores (the GPU chunk should score highest for the GPU query), and a 200-text stress-test throughput number.

## Configuration

All via environment variables — no code changes needed.

| Env var | Default | Description |
|---|---|---|
| `PORT` | `8060` | HTTP port |
| `API_KEY` | _(empty)_ | If set, all endpoints (except `/health`) require matching `X-API-Key` header |
| `MODEL_NAME` | `BAAI/bge-m3` | HuggingFace model id. You probably shouldn't change this. |
| `MAX_MEMORY_FRACTION` | `0.40` | Hard cap on GPU memory, as a fraction of total VRAM. Lower = safer for TTS, but risks OOM on big batches |
| `BATCH_SIZE` | `32` | Default batch size for the model forward pass |
| `MAX_BATCH_SIZE` | `128` | Upper bound the client is allowed to request |
| `MAX_LENGTH` | `1024` | Default token truncation. Shorter = much faster |
| `PIPELINE_DEPTH` | `4` | Max in-flight requests before returning 503 |

CLI args (override env):

```bash
python server.py --host 0.0.0.0 --port 8060 --api-key mysecret --model BAAI/bge-m3
```

## API

### `GET /health`

```json
{
  "status": "ok",
  "model": "BAAI/bge-m3",
  "device": "NVIDIA GeForce RTX 5070",
  "vram_mb": 1174.3,
  "memory_fraction_cap": 0.40
}
```

---

### `POST /embed`

Batch embed a list of texts. This is the main endpoint you'll use for both indexing and querying.

**Request:**

| Field | Type | Default | Description |
|---|---|---|---|
| `texts` | `string[]` | _(required)_ | Texts to embed |
| `normalize` | `bool` | `true` | L2-normalize vectors (leave `true` for cosine similarity) |
| `max_length` | `int` | `1024` | Token truncation (16–8192) |
| `batch_size` | `int` | `32` | Internal forward-pass batch size (1–128) |

**Response:**

```json
{
  "embeddings": [[...1024 floats...], [...], ...],
  "model": "BAAI/bge-m3",
  "dimension": 1024,
  "count": 3,
  "processing_time": 0.087
}
```

**Example:**

```bash
curl -X POST http://localhost:8060/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["first chunk", "second chunk", "third chunk"]}'
```

---

### `POST /embed/query`

Convenience endpoint for a single query text. Same model, same vectors as `/embed`, just flatter output.

**Request:**

```json
{"text": "What is BGE-M3?", "normalize": true, "max_length": 1024}
```

**Response:**

```json
{
  "embedding": [...1024 floats...],
  "dimension": 1024,
  "processing_time": 0.019
}
```

---

## RAG usage pattern

### 1. Indexing (at document ingestion time)

```python
import requests

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    resp = requests.post(
        "http://localhost:8060/embed",
        json={"texts": chunks, "max_length": 1024, "batch_size": 32},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]

# After your chunker splits a document:
chunks = ["chunk 1 text...", "chunk 2 text...", ...]
vectors = embed_chunks(chunks)

# Store in your vector DB — e.g. Qdrant:
# qdrant.upsert(collection_name="docs", points=[...])
```

Send large lists (up to a few thousand) in one call — the server internally batches them at `batch_size` at a time. Splitting into many tiny HTTP calls is strictly slower.

### 2. Querying (at search time)

```python
def embed_query(text: str) -> list[float]:
    resp = requests.post(
        "http://localhost:8060/embed/query",
        json={"text": text},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

q_vec = embed_query("user's question")
# hits = qdrant.search(collection_name="docs", query_vector=q_vec, limit=5)
```

### 3. Batching multiple queries

If you need to embed many queries at once (e.g. hybrid search with multi-query expansion), use `/embed` with a list — it's the same model, and batching is ~10× faster than serial single-query calls.

## Running as a background service

### tmux / screen

```bash
tmux new -s bge
source venv/bin/activate
python server.py --port 8060
# Ctrl-B D to detach
```

### systemd (recommended for production)

Create `/etc/systemd/system/bge-m3.service`:

```ini
[Unit]
Description=BGE-M3 Embedding Server
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/omnivoice_tts/bge_m3_server
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=MAX_MEMORY_FRACTION=0.40
Environment=PORT=8060
ExecStart=/path/to/omnivoice_tts/bge_m3_server/venv/bin/python server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bge-m3
sudo journalctl -u bge-m3 -f
```

## Sharing the GPU with the TTS server

Recommended settings on a 12 GB RTX 5070:

| Process | VRAM budget | How |
|---|---|---|
| TTS (OmniVoice) | ~7 GB | Existing config |
| BGE-M3 | ~4.8 GB | `MAX_MEMORY_FRACTION=0.40` |
| Headroom | ~200 MB | — |

If TTS is tight on memory, **lower** `MAX_MEMORY_FRACTION` (e.g. `0.30` → ~3.6 GB cap) and reduce `MAX_BATCH_SIZE` / `MAX_LENGTH` accordingly.

If you start seeing CUDA OOM in either process, check `nvidia-smi` and adjust the fraction down.

## Troubleshooting

**`CUDA out of memory` at startup**
The fraction cap is smaller than the model + activations need. Raise `MAX_MEMORY_FRACTION` to `0.45` or `0.50`.

**`CUDA out of memory` during `/embed`**
Big batch × long sequences. Lower `batch_size` or `max_length` in the request, or in the env.

**`503 Server busy, try again`**
`PIPELINE_DEPTH` is saturated. Either slow your client or raise it (but raising it too high will just queue work and blow latency).

**First request is slow (~10 s)**
Expected — that's warmup. Subsequent requests run at steady-state speed.

**Throughput is lower than expected**
Check `nvidia-smi`: if TTS is hot you'll share SM time. Also verify FP16 is active — `/health` should show a CUDA device name.

## Files

```
bge_m3_server/
├── server.py           # FastAPI server
├── test_client.py      # Example client (batch + query + stress test)
├── requirements.txt    # Python deps (bare-metal install)
├── setup.sh            # One-shot uv venv + uv pip install
├── Dockerfile          # GPU Docker image
├── docker-compose.yml  # Docker Compose (host networking, nvidia runtime)
├── .env.example        # Env var template
└── README.md           # This file
```

## Model

- **Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Dimension:** 1024 (dense)
- **Max tokens:** 8192
- **Languages:** 100+ (English, Uzbek, Russian, Chinese, etc.)
- **License:** MIT
