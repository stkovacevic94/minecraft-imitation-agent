FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
# nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install base utilities
RUN apt-get update && \
    apt-get install -y \
    openjdk-8-jdk \
    build-essential  \
    wget  \
    software-properties-common \
    xvfb \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \
    python-opengl \
    x11-xserver-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda update -y conda

# Install conda packages
RUN conda install python=3.9 pytorch=1.12.0 cudatoolkit=10.2 -c pytorch

# Install pip packages
RUN pip3 install wandb minerl