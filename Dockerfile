FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENV=/.venv

RUN pip3 install uv

COPY ./requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.11 ${UV_PROJECT_ENV}
# Note that syncing will install only the packages listed in requirements.txt
# Running `uv run python3` will install the other dependencies supporting said packages
RUN uv pip sync /tmp/requirements.txt

WORKDIR /app