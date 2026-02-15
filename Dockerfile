# =============================================================================
# Vaeda Development Container (v0.2.0 — PyTorch backend)
# =============================================================================
# Build targets:
#   Production:   docker build --target production -t vaeda:prod .
#   Development:  docker build --target development -t vaeda:dev .
#
# Run examples:
#   docker run -it --rm -v $(pwd):/usr/vaeda vaeda:dev
#   docker compose up dev
# =============================================================================

# -----------------------------------------------------------------------------
# Base stage: common setup for all targets
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /usr/vaeda \
    && mkdir -p /root/.ssh && chmod 700 /root/.ssh

RUN command -v uv >/dev/null 2>&1 || \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /usr/vaeda

# uv configuration
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# -----------------------------------------------------------------------------
# Production stage: core dependencies only
# -----------------------------------------------------------------------------
FROM base AS production

COPY pyproject.toml uv.lock* ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project 2>/dev/null || \
    uv sync --no-dev --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT []
CMD ["sh", "-c", "python -c \"import vaeda; print(f'vaeda {vaeda.__version__} prod ready (PyTorch backend)')\" && sleep infinity"]

# -----------------------------------------------------------------------------
# Development stage: core + dev dependencies
# -----------------------------------------------------------------------------
FROM base AS development

RUN apt-get update && apt-get install -y --no-install-recommends \
    ripgrep \
    fd-find \
    fzf \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --no-install-project 2>/dev/null || \
    uv sync --group dev --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev 2>/dev/null || uv sync --group dev

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 80

ENTRYPOINT []
CMD ["sh", "-c", "python -c \"import vaeda; print(f'vaeda {vaeda.__version__} dev ready (PyTorch backend)')\" && sleep infinity"]
