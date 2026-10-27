# Reproduction of submission 046 v4. Python 3.11 + uv; torch 2.7.1 (CUDA 12.8 wheels) comes from uv.lock.
# The CUDA runtime libraries ship inside the torch wheel; the host only provides the NVIDIA driver via the
# NVIDIA container runtime (compose: deploy.resources.reservations.devices).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgomp: lightgbm / xgboost / scikit-image; libgl / libglib: opencv (albumentations); git: the tracksdata dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/solution

# dependencies first (layer cache), then the package
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY biohub046 ./biohub046
COPY third_party ./third_party
COPY tests ./tests
COPY run.sh README.md ./
RUN uv sync --frozen --no-dev

ENV BIOHUB_INPUT_DIR=/work/input \
    BIOHUB_ARTIFACTS_DIR=/work/artifacts \
    BIOHUB_CACHE_DIR=/work/cache \
    HF_HOME=/work/cache/hf \
    UV_CACHE_DIR=/work/cache/uv \
    MPLCONFIGDIR=/work/cache/mpl \
    NO_ALBUMENTATIONS_UPDATE=1

ENTRYPOINT ["./run.sh"]
CMD ["all"]
