FROM ubuntu:24.04

# Taken from https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#sort-multi-line-arguments
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  gpg \
  software-properties-common \
  sudo \
  wget \
  && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
  && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
  && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
  && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cmake \
  file \
  git \
  golang-go \
  ninja-build \
  sudo \
  vim-nox \
  bash-completion \
  gdb \
  build-essential \
  ccache \
  clang-20 \
  mold \
  llvm-20 \
  libmlir-20-dev \
  mlir-20-tools \
  pkg-config \
  zlib1g-dev \
  libzstd-dev \
  && rm -rf /var/lib/apt/lists/*

# - Use clang-20 as default compiler
# - Allow user 'ubuntu' to run commands as root
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-20 100 \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod a=r /etc/sudoers.d/ubuntu

# Debug build of LLVM
# ADD build_llvm_debug.sh /tmp/build_llvm_debug.sh
# RUN /tmp/build_llvm_debug.sh && rm /tmp/build_llvm_debug.sh

USER ubuntu
WORKDIR /home/ubuntu
