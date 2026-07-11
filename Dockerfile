# syntax=docker/dockerfile:1

ARG PY_IMAGE=python:3.14-slim-trixie

FROM ${PY_IMAGE} AS base

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app


FROM base AS builder

# Live-trading image: postgres/ai/exchange only -- optuna+sqlalchemy stay out
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra postgres --extra ai --extra exchange

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra postgres --extra ai --extra exchange --no-editable


FROM ${PY_IMAGE} AS runtime

LABEL org.opencontainers.image.source="https://github.com/davidkvasnovsky/dojiwick2"

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

USER appuser

ENTRYPOINT ["dojiwick"]
CMD ["run", "--config", "config.toml"]


FROM base AS tooling

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=arigaio/atlas:1.2.3-community /atlas /usr/local/bin/atlas

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Editable install resolves /app/src from the compose bind mount; never re-sync at run time.
ENV UV_NO_SYNC=1

CMD ["sleep", "infinity"]
