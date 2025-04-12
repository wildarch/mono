FROM ghcr.io/wildarch/mono:ci

# Debug build of LLVM
USER root
ADD build_llvm_debug.sh /tmp/build_llvm_debug.sh
RUN /tmp/build_llvm_debug.sh && rm /tmp/build_llvm_debug.sh

USER ubuntu
WORKDIR /home/ubuntu
