#!/usr/bin/env bash
# Host prerequisites on Ubuntu (tested on 26.04): Docker Engine + Compose v2 from the Ubuntu archive and the
# NVIDIA Container Toolkit from NVIDIA's apt repository, then register the nvidia runtime with Docker.
# Run as root:  sudo bash solution/scripts/install_docker_ubuntu.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y docker.io docker-compose-v2 curl gpg

install -d -m 0755 /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

if [ -n "${SUDO_USER:-}" ]; then
  usermod -aG docker "$SUDO_USER"
  echo "added $SUDO_USER to the docker group (log in again, or run: newgrp docker)"
fi
docker --version
docker compose version
docker info 2>/dev/null | grep -i runtimes
