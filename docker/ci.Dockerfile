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
  && wget https://repo1.maven.org/maven2/org/apache/arrow/ubuntu/apache-arrow-apt-source-latest-noble.deb \
  && apt install -y -V ./apache-arrow-apt-source-latest-noble.deb \
  && rm apache-arrow-apt-source-latest-noble.deb \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  bash-completion \
  build-essential \
  ccache \
  clang-20 \
  cmake \
  file \
  gdb \
  git \
  golang-go \
  libarrow-dev \
  libmlir-20-dev \
  libparquet-dev \
  libzstd-dev \
  llvm-20 \
  mlir-20-tools \
  mold \
  ninja-build \
  pkg-config \
  protobuf-compiler \
  python3-pip \
  sudo \
  vim-nox \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages lit

# - Use clang-20 as default compiler
# - Allow user 'ubuntu' to run commands as root
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-20 100 \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod a=r /etc/sudoers.d/ubuntu

# External dependencies
WORKDIR /opt

# libpg_query
RUN wget -O - https://github.com/pganalyze/libpg_query/archive/refs/tags/17-6.1.0.tar.gz | tar xzf - \
    && make -C libpg_query-17-6.1.0/ install
