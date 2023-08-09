# Nvidia

## Prerequisites

**Install docker** https://docs.docker.com/engine/install/

**Install Nvidia driver** You can check with `nvidia-smi` whether drivers are already installed.

**Install nvidia-docker2** e.g. `dnf install nvidia-docker2` or `yay -Sy nvidia-docker`

## Build Image

```
docker build -f ./hip-libraries-cuda-ubuntu.Dockerfile -t hip_nvidia .
```

## Start container

When creating a new container you have to pass through the gpu this is done using the `--gpus all` argument. It is also possible to only pass through a specific GPU for more information on how to do that refer to the nvidia-docker2 documentation.

**Example**
```
docker run -it --gpus all --name hip_docker --network host hip_nvidia /bin/bash
```

## Limitations

At the moment, there is an issue with dependencies that is hindering installations through `apt`. If you require any extra software, it is best to install it at the start of the docker file.

# AMD

## Prerequisites

**Install docker** https://docs.docker.com/engine/install/

**Install AMD driver** If `/dev/kfd` and `/dev/dri` already exist, it is likely that the drivers are already installed.

## Build Image

```
docker build -f ./hip-libraries-rocm-ubuntu.Dockerfile -t hip_amd .
```

## Start Container

When creating a new container you have to pass through the GPU this is done using the `--device=/dev/kfd --device=/dev/dri` arguments. It is also possible to only pass through a specific GPU for more information on how to do that refer to the ROCM documentation.

**Example**
```
docker run -it --device=/dev/kfd --device=/dev/dri --name rocm_docker_2 --network host hip_docker /bin/bash
```