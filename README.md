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

- **Effortless Deployment:** Quickly deploy the chatbot using Docker. Simply run a few commands or use the provided scripts.
- **Self-Improvement:** The chatbot learns and adapts from conversations, delivering context-aware and engaging responses.
- **GPU Acceleration:** Boost your chatbot's performance with GPU support (NVIDIA's CUDA or AMD's ROCm) for advanced users.

## Getting Started

1. **Install Dependencies:**
   - **For General Usage (CPU):**
     Open a terminal and run the following command to install Docker and its dependencies:
     ```bash
     ./GPUdeploy_bot.sh
     ```

   - **For NVIDIA GPUs (CUDA):**
     Open Command Prompt and execute the following command to build and run the chatbot container with Numba and CUDA support:
     ```bat
     Deploy_bot_windows.bat
     ```

   - **For AMD GPUs (ROCm Support):**
     Run the following command in a terminal to install ROCm dependencies and configure Docker for AMD GPU acceleration:
     ```bash
     ./DockerRoCMDeploy_bot.sh
     ```

2. **Build and Launch the Docker Container:**
   Navigate to the directory with the Dockerfile and run the following commands:
   ```bash
   docker build -t bot .
   docker run -it --rm --name bot_container bot
Engage in Conversations:
Once the container is running, use the included Python script bot.py to simulate conversations and observe the chatbot's self-improvement. In the container's terminal, run:
bash
Copy code
python bot.py simulate_conversation
Use Cases
Customer Support: Automate responses to common queries for swift customer service.
Language Learning: Facilitate language learning via simulated conversations.
Entertainment: Craft interactive and amusing chatbot experiences.
Research: Experiment with AI and NLP techniques for research and analysis.
Contributing
Contributions are welcomed! Feel free to fork this repository, enhance, and submit pull requests. If you come across any issues or have suggestions, please open an issue.

Credits
Developed by [Your Name]. Inspired by advancements in AI and NLP.

Disclaimer
This chatbot is designed for educational and experimental purposes. It lacks real-world intelligence. Use AI responsibly and consider ethical implications.

<p align="center">
  Made with ❤️ by <a href="https://zchg.org">zchg.org</a>
</p>
```
This version includes the instructions for launching the Docker container using the bot.py script within the container to simulate conversations.
![image](https://github.com/ZCHGorg/ChatAI/assets/24325826/89bc91e7-8959-4607-a7bd-bd8f4b115645)

