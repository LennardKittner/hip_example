# Nvidia

## Prerequisites

**install docker** https://docs.docker.com/engine/install/

**install nvidia driver** You can check with `nvidia-smi` whether drivers are already installed.

**install nvidia-docker2** `dnf install nvidia-docker2` `yay -Sy nvidia-docker`

## Build Image

```
docker build -f ./hip-libraries-cuda-ubuntu.Dockerfile -t hip_nvidia .
```

## Start container

When creating a new container you have to pass through the gpu this is done using the `--gpus all` argument. It is also possible to only pass through a specific gpu for more information on how to do that refer to the nvidia-docker2 documentation.

**Example**
```
docker run -it --gpus all --name hip_docker --network host hip_nvidia /bin/bash
```

# AMD

## Prerequisites

**install docker** https://docs.docker.com/engine/install/

**install amd driver** The drivers are probably already installed if `/dev/kfd` and `/dev/dri` already exist.

## Build Image

```
docker build -f ./hip-libraries-rocm-ubuntu.Dockerfile -t hip_amd .
```

## Start Container

When creating a new container you have to pass through the gpu this is done using the `--device=/dev/kfd --device=/dev/dri` arguments. It is also possible to only pass through a specific gpu for more information on how to do that refer to the rocm documentation.

**Example**
```
docker run -it --device=/dev/kfd --device=/dev/dri --name rocm_docker_2 --network host hip_docker /bin/bash
```