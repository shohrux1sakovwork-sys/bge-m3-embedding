FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    "FlagEmbedding>=1.2.10" \
    "transformers>=4.40,<5" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.0" \
    "orjson"

COPY server.py .

ENV PORT=8060
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8060

CMD ["python", "server.py", "--host", "0.0.0.0"]
