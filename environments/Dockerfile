FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10.20
ARG IMITATOR_GAME_REVISION=d6d16ec511bc389e0a207692730c137bc022ef14
ENV LANG=C.UTF-8 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git pkg-config cmake ninja-build \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev libgdbm-dev libncursesw5-dev uuid-dev ffmpeg \
    libgl1 libegl1 libglvnd0 libvulkan1 vulkan-tools libx11-6 libxext6 \
    libxrender1 libxi6 libsm6 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
    | tar -xz -C /tmp && cd "/tmp/Python-${PYTHON_VERSION}" \
    && ./configure --prefix=/opt/python --with-ensurepip=install \
    && make -j4 && make install && rm -rf "/tmp/Python-${PYTHON_VERSION}"
ENV PATH=/opt/python/bin:${PATH}
RUN ln -s /opt/python/bin/python3 /opt/python/bin/python \
    && python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0
RUN python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
COPY requirements-evaluation.txt /opt/robodemo-environment/requirements-evaluation.txt
RUN python -m pip install -r /opt/robodemo-environment/requirements-evaluation.txt
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MAX_JOBS=4
RUN python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
# Import the evaluated simulator source, not its unrelated training extras.
RUN git clone https://github.com/imitator-game/The-Imitator-Game.git /opt/ImitatorGame \
    && git -C /opt/ImitatorGame checkout "${IMITATOR_GAME_REVISION}"
COPY check_environment.py /opt/robodemo-environment/check_environment.py
COPY entrypoint.sh /opt/robodemo-environment/entrypoint.sh
ENV IMITATOR_GAME_ROOT=/opt/ImitatorGame PYTHONPATH=/opt/ImitatorGame \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
WORKDIR /workspace/RoboDemo
ENTRYPOINT ["bash", "/opt/robodemo-environment/entrypoint.sh"]
CMD ["python", "/opt/robodemo-environment/check_environment.py"]
