FROM ubuntu:jammy 

# Taken from https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#sort-multi-line-arguments
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ansible \
  git \
  gpg \
  && rm -rf /var/lib/apt/lists/*

COPY . /mono

WORKDIR "/mono"
ENTRYPOINT ["ci/run.sh"]