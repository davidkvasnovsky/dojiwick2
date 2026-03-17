FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

FROM base AS runtime_deps
RUN uv sync --locked --no-dev --extra postgres

FROM base AS qa_deps
RUN uv sync --locked --extra postgres

FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=runtime_deps /opt/venv /opt/venv
COPY . .

USER appuser

CMD ["python", "-m", "dojiwick.interfaces.cli.runner", "--config", "config.toml"]

FROM python:3.14-slim AS tooling

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=qa_deps /opt/venv /opt/venv
COPY --from=arigaio/atlas:latest /atlas /usr/local/bin/atlas
COPY . .

CMD ["sleep", "infinity"]
