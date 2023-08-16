Distributed Chat Bot with CUDA and ROCm Support
Distributed Chat Bot

Welcome to the Distributed Chat Bot! This powerful conversational AI system is designed to simulate engaging conversations while leveraging the capabilities of both NVIDIA GPUs with CUDA and AMD GPUs with ROCm. Whether you're interested in natural language processing, machine learning, or simply having fun conversations, this bot has you covered.

Features
Harness the power of CUDA and ROCm to enhance conversation simulations.
Seamless integration with Docker for easy deployment and isolation.
Utilizes Python's advanced libraries, including NLTK, scikit-learn, and more.
Leverages Levenshtein distance and other techniques for context improvement.
Supports both Windows and Linux environments.
Getting Started
Follow the steps below to set up and deploy the Distributed Chat Bot on your system:

Docker Installation
For both Windows and Linux systems, the first step is to install Docker. Choose the appropriate installer for your platform:

Docker Desktop for Windows
Docker CE for CentOS
Windows Deployment
Open the Deploy_bot_windows.bat script.
Install Docker Desktop if prompted.
Wait for the Docker Desktop installation to complete.
The script will build the Docker container and run the Distributed Chat Bot with ROCm support for AMD GPUs.
Enjoy engaging conversations with the bot!
Linux GPU Deployment
Open the GPUdeploy_bot.sh script in a terminal.
Run the script to install Docker and its dependencies.
The script will create a Docker container with the necessary Python dependencies and your bot's script.
Start the Docker container to engage in conversations with the Distributed Chat Bot.
ROCm Deployment (AMD GPUs)
If you have AMD GPUs and want to leverage ROCm, follow these additional steps:

Open the DockerRoCMDeploy_bot.sh script.
The script installs the ROCm rock-dkms kernel modules and Docker dependencies.
It configures Docker with the overlay2 storage driver for ROCm compatibility.
Build the Docker container and run the Distributed Chat Bot for ROCm-enhanced conversations.
Conversational AI at Your Fingertips
With the Distributed Chat Bot, you can explore the realms of AI-driven conversations, experiment with different GPU technologies, and enjoy meaningful interactions. Unlock the potential of CUDA and ROCm to simulate engaging and dynamic dialogues.

For questions, feedback, or contributions, please contact [Your Contact Email].

Note: The Distributed Chat Bot is a demonstration project and may require adjustments based on your system configuration. Make sure to review and adapt the provided scripts to your environment.
