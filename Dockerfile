FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update && \
    apt-get -qq install -y build-essential wget software-properties-common && \
    add-apt-repository ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get -qq install openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
