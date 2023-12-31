#!/bin/bash

# Install Docker and its dependencies
echo "Installing Docker and its dependencies..."
sudo yum update -y
sudo yum install -y epel-release
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y --allowerasing docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Create Dockerfile
echo "Creating Dockerfile..."
cat <<EOF > Dockerfile
# Use a base image with Python and necessary dependencies
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the bot's Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bot's Python script into the container
COPY self_improving_bot.py .

# Run the bot's script when the container starts
CMD ["python", "self_improving_bot.py"]
EOF

# Create requirements.txt
echo "Creating requirements.txt..."
echo "
scikit-learn
numpy
openai
" > requirements.txt

# Build Docker container
echo "Building Docker container..."
docker build -t self_improving_bot -f Dockerfile .

# Run Docker container
echo "Starting Docker container..."
docker run -it --rm --name self_improving_bot_container self_improving_bot

echo "Script completed."
