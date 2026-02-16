
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    llvm-dev \
    libtool-bin \
    automake \
    bison \
    flex \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*



ARG AFLPP_VERSION=v4.35c

WORKDIR /opt

RUN git clone --depth 1 --branch ${AFLPP_VERSION} https://github.com/AFLplusplus/AFLplusplus.git && \
    cd AFLplusplus && \
    make -j$(nproc) && \
    make install

ENV AFL_PATH=/opt/AFLplusplus

# --------------------------------------------------------------------
# Install uv (Python package manager)
# --------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# --------------------------------------------------------------------
# Python Environment Setup
# --------------------------------------------------------------------
WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY requirements.in requirements-cuda.txt uv.lock ./

RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install --upgrade pip && \
    uv pip install -r requirements.in && \
    uv pip install -r requirements-cuda.txt

ENV PATH="/app/.venv/bin:${PATH}"

# --------------------------------------------------------------------
# Copy Project Source
# --------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------
# Build Custom AFL++ Mutator
# --------------------------------------------------------------------
WORKDIR /app/aflpp-plugins
RUN make

WORKDIR /app

# --------------------------------------------------------------------
# Default Entry
# --------------------------------------------------------------------
CMD ["bash"]
