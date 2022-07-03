FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update && \
    apt-get -qq install -y build-essential wget software-properties-common && \
    add-apt-repository ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get -qq install openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install wandb pandas minerl pytorch-lightning