FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
      python3 python3-pip python3-venv git \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg

RUN pip3 install --no-cache-dir uv

ENV VENV_PATH=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=${VENV_PATH}
ENV PATH=${VENV_PATH}/bin:$PATH       
ENV VIRTUAL_ENV=${VENV_PATH}          

WORKDIR /app          

COPY pyproject.toml uv.lock* ./

RUN uv venv --python 3.11 \
 && (uv sync --locked || uv sync)     