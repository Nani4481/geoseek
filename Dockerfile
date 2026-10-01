# geoseek - offline-first geospatial ML platform.
#
# CPU-only by default (portable: no host CUDA/nvidia-container-toolkit
# required). GPU serving would require a separately validated CUDA image;
# adding --gpus alone cannot enable CUDA in this CPU-only torch build.
#
# `data/` is intentionally NOT baked into the image - it holds staged model
# weights and the ingested tile/index catalog (gigabytes, machine-specific,
# already gitignored) and is expected to be bind-mounted at run time. See
# RUN.md for how to stage it the first time; docker-compose.yml mounts
# ./data -> /app/data by default.
FROM python:3.11-slim

# libgl1/libglib2.0-0: opencv-python needs these at import time even in a
# headless container (it links against them regardless of display use).
# libgomp1: OpenMP runtime faiss-cpu and torch both call into.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies against a package *stub* before the real source is
# copied in, so `pip install` only re-runs (and re-downloads every heavy
# wheel: torch, opencv, faiss, rasterio, ...) when pyproject.toml itself
# changes, not on every source edit. This works because the install below
# is editable (-e): it links the import path back to /app/src rather than
# copying files into site-packages, so the later `COPY . .` makes the real
# modules importable with no reinstall.
COPY pyproject.toml ./
RUN mkdir -p src/geoseek && touch src/geoseek/__init__.py

# CPU-only torch build - the default pip resolution on Linux already
# prefers this, pinned explicitly here so the image never silently pulls a
# multi-gigabyte CUDA wheel it can't use without --gpus.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
ARG GEOSEEK_EXTRAS=detect
RUN if [ -n "$GEOSEEK_EXTRAS" ]; then pip install --no-cache-dir -e ".[${GEOSEEK_EXTRAS}]"; else pip install --no-cache-dir -e .; fi

COPY . .

# Pre-create the data layout config.py expects (ensure_dirs() would do this
# too at first import, but doing it at build time means an empty bind mount
# still has the right skeleton before the app ever runs).
RUN mkdir -p data/models data/datasets data/tiles data/index

EXPOSE 8000

# 0.0.0.0, not the 127.0.0.1 loopback RUN.md uses for a bare-metal run: the
# offline guarantee is about outbound network calls the app itself makes
# (enforced by tests/test_frontend_offline.py and friends), not about which
# interface it listens on - and 127.0.0.1 *inside* the container would be
# unreachable from the host through Docker's own port mapping.
ENV GEOSEEK_DEVICE=cpu PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
CMD ["sh", "-c", "exec uvicorn geoseek.search.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
