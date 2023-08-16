@echo off

:: Install Docker Desktop
echo Installing Docker Desktop...
start https://www.docker.com/products/docker-desktop

:: Wait for Docker Desktop installation to complete
echo Waiting for Docker Desktop installation to complete...
timeout /t 60

:: Build Docker container
echo Building Docker container...
docker build -t self_improving_bot -f Dockerfile .

:: Run Docker container with ROCm support (for AMD GPUs)
echo Starting Docker container with ROCm support...
docker run --rm -it --name self_improving_bot_container_rocm --device=/dev/kfd --device=/dev/dri --group-add video self_improving_bot

:: Run Docker container with Numba and CUDA support (for NVIDIA GPUs)
echo Starting Docker container with Numba and CUDA support...
docker run --rm -it --name self_improving_bot_container_cuda --gpus all self_improving_bot

echo Script completed.
