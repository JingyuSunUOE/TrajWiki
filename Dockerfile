FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
LABEL maintainer="jingyu sun"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TRAJWIKI_GIT_COMMIT=unknown
ARG TRAJWIKI_GIT_DIRTY=unknown
ARG TRAJWIKI_SOURCE_HASH=unknown
ENV TRAJWIKI_GIT_COMMIT=${TRAJWIKI_GIT_COMMIT}
ENV TRAJWIKI_GIT_DIRTY=${TRAJWIKI_GIT_DIRTY}
ENV TRAJWIKI_SOURCE_HASH=${TRAJWIKI_SOURCE_HASH}

# 安装系统依赖与 Python 3.11
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        git \
        git-lfs \
        curl \
        nvtop \
        screen \
        ca-certificates \
        libsndfile1-dev \
        libgl1 \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 固定构建工具，避免未来 uv 解析行为变化影响正式实验镜像。
RUN python -m pip install --no-cache-dir --upgrade pip "uv==0.9.5"

WORKDIR /opt/trajwiki

# 先只复制依赖描述文件
COPY pyproject.toml ./

# 在一次解析中安装全部运行依赖。vLLM 0.10.2 会约束 PyTorch 2.8，
# --torch-backend cu126 确保解析到 CUDA 12.6 wheel，而不是 CPU wheel。
RUN uv pip install --no-cache --strict \
    --torch-backend cu126 \
    -r pyproject.toml

# 环境变量
ENV TORCH_HOME=/data/users/jingyu/.torch
ENV HF_HOME=/data/users/jingyu/Huggingface
ENV HF_DATASETS_CACHE=/data/users/jingyu/Huggingface
ENV DNNLIB_CACHE_DIR=/data/users/jingyu/.cache/dnnlib
ENV TORCH_EXTENSIONS_DIR=/data/users/jingyu/.torch/torch_extensions
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"
ENV PYTORCH_KERNEL_CACHE_PATH=/data/users/jingyu/.torch/kernels
ENV UV_CACHE_DIR=/data/users/jingyu/.cache

# 安装当前源码，并保留源码树供实验 manifest 计算可复现哈希。
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv pip install --no-cache --no-deps . \
    && chmod +x scripts/*.sh \
    && python -m pip check \
    && python -c "from importlib.metadata import version; import trajpatch; packages = ('trajwiki', 'torch', 'transformers', 'sentence-transformers', 'vllm', 'tiktoken', 'zstandard', 'openai', 'anthropic'); print({name: version(name) for name in packages})"
ENV PYTHONPATH="/opt/trajwiki/src"

# 添加一个用户

ARG USER_ID=35761
ARG GROUP_ID=4451
RUN groupadd -g ${GROUP_ID} usergroup && \
    useradd -m -u ${USER_ID} -g usergroup jingyu
USER ${USER_ID}

# 端口与工作目录
EXPOSE 8081
