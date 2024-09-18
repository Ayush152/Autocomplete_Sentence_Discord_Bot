# Discord Sentence Completion Bot using N-gram Language Models

This repository contains a Python-based Discord bot that automatically completes user-provided phrases using bigram and trigram language models trained on Shakespeare's texts (or any provided corpus). The bot implements Laplace smoothing and backoff techniques to handle unseen word sequences, ensuring coherent sentence generation.

## Features

- **Discord Bot Integration**: Built using the `discord.py` library, the bot responds to user messages, generating the next 10 words based on the input phrase.
- **N-gram Language Models**: Includes unigram, bigram, and trigram models built from the training data.
- **Laplace Smoothing**: Applies add-one smoothing to avoid zero probabilities for unseen bigrams or trigrams.
- **Backoff Mechanism**: The trigram model falls back to the bigram model, and further to the unigram model when necessary, ensuring robust predictions.
- **Weighted Word Selection**: Predicts the next word by selecting more probable word pairs or triples based on their frequency in the training data.
- **Log Probability Calculation**: Displays log probabilities for the generated sentences, indicating the likelihood of the predicted sentence given the input.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/discord-ngram-bot.git
2. Make a virtual environment
   ```bash
   python -m venv /path/to/new/virtual/environment
3. Install the required dependencies:
   ```bash
   pip install discord.py
4. Make a .env file with 2 key values
   ```bash
   DISCORD_TOKEN=YOUR_DISCORD_TOKEN
   TRAINING_DATA=Shakespeare.txt
5. Run the bot
   ```bash
   python main.py

## Usage

- Invite the bot to your Discord server.
- Type a message <your phrase> in a channel where the bot is active.
- The bot will generate the next 10 words based on both bigram and trigram models, along with their log probabilities.
