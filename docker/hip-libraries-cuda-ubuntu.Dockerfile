# Source: https://github.com/amd/rocm-examples/blob/develop/Dockerfiles/hip-libraries-cuda-ubuntu.Dockerfile
# CUDA based docker image
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Base packages that are required for the installation
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    locales-all \
    make \
    python3 \
    ssh \
    sudo \
    wget \
    pkg-config \
    glslang-tools \
    libvulkan-dev \
    vulkan-validationlayers \
    libglfw3-dev \
    gfortran \
    # Nvidia driver version needed for hipSOLVER's CUDA backend.
    # See https://docs.nvidia.com/deploy/cuda-compatibility/index.html#default-to-minor-version.
    nvidia-driver-455 \
    && rm -rf /var/lib/apt/lists/*

# Install HIP using the installer script
RUN export DEBIAN_FRONTEND=noninteractive; \
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.3/ ubuntu main' > /etc/apt/sources.list.d/rocm.list \
    && apt-get update -qq \
    && apt-get install -y hip-base hipify-clang \
    && apt-get download hip-runtime-nvidia hip-dev \
    && dpkg -i --ignore-depends=cuda hip*

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.7/cmake-3.21.7-linux-x86_64.sh \
    && mkdir /cmake \
    && sh cmake-3.21.7-linux-x86_64.sh --skip-license --prefix=/cmake \
    && rm cmake-3.21.7-linux-x86_64.sh

ENV PATH="/cmake/bin:/opt/rocm/bin:${PATH}"

RUN echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
    && ldconfig