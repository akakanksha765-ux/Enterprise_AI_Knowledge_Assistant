# -------------------------------------------------------
# Enterprise AI Knowledge Assistant
# -------------------------------------------------------

FROM python:3.11-slim

# Prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

# Working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies (cache downloaded packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy project
COPY . .

# FastAPI port
EXPOSE 8000

# Start API
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]