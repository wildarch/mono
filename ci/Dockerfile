FROM ubuntu:24.04 

# Taken from https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#sort-multi-line-arguments
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ansible \
  git \
  gpg \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# Allow executing commands as root using sudo
RUN echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# Bazel does not like running as root
USER ubuntu

COPY . /home/ubuntu/mono

WORKDIR "/home/ubuntu/mono"
ENTRYPOINT ["ci/run.sh"]