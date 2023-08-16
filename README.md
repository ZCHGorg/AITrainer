<h1 align="center">
  <br>
  <img src="chatbot_logo.png" alt="Distributed Chatbot" width="200">
  <br>
  Distributed Chatbot with Docker
  <br>
</h1>

<p align="center">
  <strong>Simulate Conversations with GPU Power</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#credits">Credits</a>
</p>

<p align="center">
  <img src="chatbot_demo.gif" alt="Distributed Chatbot Demo">
</p>

## Features

- **Effortless Deployment:** Easily deploy the chatbot using Docker. Use the command `docker build -t bot . && docker run -it bot` to get started.
- **Self-Improvement:** The chatbot learns and adapts from conversations, providing more engaging and context-aware responses.
- **GPU Acceleration:** Accelerate your chatbot's performance with GPU support for advanced users (NVIDIA's CUDA or AMD's ROCm).

## Getting Started

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop).

2. Choose Your Deployment:

   - **For General Usage (CPU):**
     Run `GPUdeploy_bot.sh` to easily install Docker and build/run the chatbot container.

   - **For NVIDIA GPUs (CUDA):**
     Execute `Deploy_bot_windows.bat` to leverage Numba and CUDA support for GPU acceleration.

   - **For AMD GPUs (ROCm Support):**
     Run `DockerRoCMDeploy_bot.sh` to install ROCm dependencies and configure Docker for AMD GPU acceleration.

3. Engage in Conversations:
   Once the container is running, use the provided Python script `bot.py` to simulate conversations and witness the chatbot's self-improvement.

## Use Cases

- **Customer Support:** Automate common queries and provide instant responses.
- **Language Learning:** Facilitate language learning through simulated conversations.
- **Entertainment:** Create interactive and entertaining chatbot experiences.
- **Research:** Experiment and explore AI and NLP techniques.

## Contributing

Contributions are welcome! Fork this repository, make enhancements, and submit pull requests. If you encounter issues or have suggestions, open an issue.

## Credits

Developed by [Your Name]. Inspired by AI and NLP advancements.

## Disclaimer

This chatbot is intended for educational and experimental purposes. It lacks real-world intelligence. Use AI responsibly and consider ethical implications.

---

<p align="center">
  Made with ❤️ by https://zchg.org
</p>
