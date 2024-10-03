FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev

WORKDIR /app

COPY pyproject.toml /app/
RUN pip3 install --no-cache-dir pyproject.toml

COPY . /app

EXPOSE 8000
