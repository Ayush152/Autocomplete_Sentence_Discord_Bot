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
2. Install the required dependencies:
  pip install -r requirements.txt
3. Add your Discord Bot Token to the environment or directly in the script:
  ```bash
  client.run('YOUR_DISCORD_API_TOKEN')
