# --- builder: install dependencies into a discardable layer ---
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml README.md ./
# CPU-only torch index first, so pip resolves the small CPU wheel instead of pulling CUDA deps.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# --- runtime: copy only installed packages + app code, no build cache ---
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app ./app
COPY etl ./etl
COPY ml ./ml

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
