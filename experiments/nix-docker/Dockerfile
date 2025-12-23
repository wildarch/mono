FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  xz-utils \
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ADD https://nixos.org/nix/install /nix-install
RUN chmod +x /nix-install && chown ubuntu:ubuntu /nix-install

RUN mkdir -m 0755 /nix && chown ubuntu /nix

USER ubuntu
WORKDIR /home/ubuntu

RUN /nix-install

# Need to set this for the next environment to properly load.
ENV USER=ubuntu