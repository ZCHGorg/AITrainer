# Use a base image with Python and necessary dependencies
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the bot's Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openai   # Install the openai package

# Copy your bot's Python script into the container
COPY self_improving_bot.py .
COPY conversation_library.py .

# Run the bot's script when the container starts
CMD ["python", "self_improving_bot.py"]
