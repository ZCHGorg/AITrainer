<h1 align="center">
  <br>
  <img src="chatbot_logo.png" alt="Distributed Chatbot" width="200">
  <br>
  Distributed Machine Learning Swarm Bot with Docker
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

- **Effortless Deployment:** Quickly deploy the chatbot using Docker. Just use the command `docker build -t bot . && docker run -it bot`.
- **Self-Improvement:** The chatbot learns and adapts from conversations, delivering context-aware and engaging responses.
- **GPU Acceleration:** Boost your chatbot's performance with GPU support (NVIDIA's CUDA or AMD's ROCm) for advanced users.

## Getting Started

1. **Install [Docker Desktop](https://www.docker.com/products/docker-desktop).**

2. **Choose Your Deployment:**

   - **For General Usage (CPU):**
     Open a terminal and run `GPUdeploy_bot.sh` to quickly install Docker and build/run the chatbot container.

   - **For NVIDIA GPUs (CUDA):**
     Open Command Prompt and execute `Deploy_bot_windows.bat` to leverage Numba and CUDA for accelerated GPU support.

   - **For AMD GPUs (ROCm Support):**
     Run `DockerRoCMDeploy_bot.sh` in a terminal to install ROCm dependencies and configure Docker for AMD GPU acceleration.

3. **Engage in Conversations:**
   Once the container is running, use the included Python script `bot.py` to simulate conversations and observe the chatbot's self-improvement.

## Use Cases

- **Customer Support:** Automate responses to common queries for swift customer service.
- **Language Learning:** Facilitate language learning via simulated conversations.
- **Entertainment:** Craft interactive and amusing chatbot experiences.
- **Research:** Experiment with AI and NLP techniques for research and analysis.

## Contributing

Contributions are welcomed! Feel free to fork this repository, enhance, and submit pull requests. If you come across any issues or have suggestions, please open an issue.

## Credits

Developed by [Your Name]. Inspired by advancements in AI and NLP.

## Disclaimer

This chatbot is designed for educational and experimental purposes. It lacks real-world intelligence. Use AI responsibly and consider ethical implications.

---

<p align="center">
  Made with ❤️ by <a href="https://zchg.org">zchg.org</a>
</p>
