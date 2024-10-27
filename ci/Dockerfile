FROM ubuntu:jammy 

# Taken from https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#sort-multi-line-arguments
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ansible \
  git \
  gpg \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# Bazel does not like running as root
RUN useradd -ms /bin/bash ubuntu

# Allow executing commands as root using sudo
RUN echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

COPY . /home/ubuntu/mono

WORKDIR "/home/ubuntu/mono"


USER ubuntu
ENTRYPOINT ["ci/run.sh"]