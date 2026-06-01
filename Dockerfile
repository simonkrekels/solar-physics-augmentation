FROM python:3.14-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies before copying source so this layer is cached on code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY augmentation/ augmentation/
COPY training/ training/
COPY api/ api/

EXPOSE 8000

# Checkpoint must be mounted at runtime:
#   docker run -v $(pwd)/checkpoints:/app/checkpoints -p 8000:8000 solar-thermal-cv
ENV CHECKPOINT=checkpoints/physics-augmented_best.pt

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
