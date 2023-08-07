# Use a base image with Python and necessary dependencies
FROM python:3.8-slim-buster

#RUN pip install spacy==3.1.3
#RUN python -m spacy download en_core_web_md

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
COPY conversation_log.txt .

RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install openai   # Install the openai package
RUN pip install Levenshtein

# Install NLTK and download required datasets
RUN pip install nltk
RUN python -m nltk.downloader words brown

# Set NLTK_DATA to the current directory
ENV NLTK_DATA=/

# Copy your bot's Python script into the container
COPY self_improving_bot.py .

# Run the bot's script when the container starts
CMD ["python", "self_improving_bot.py", "simulate_conversation"]