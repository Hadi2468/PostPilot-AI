FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,dashboard]"

COPY dashboard ./dashboard
COPY data ./data

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000 8501

# Default: the API. docker-compose overrides the command for the dashboard.
CMD ["uvicorn", "postpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
