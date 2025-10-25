# Stock News Sentiment Analysis

A project to fetch financial news from the Alpha Vantage API and analyze its sentiment using two distinct methods: the lexicon-based VADER (NLTK) and the specialized Transformer model, FinBERT.

## Features

* **News Extraction**: Connects to the Alpha Vantage API to fetch news by tickers (e.g., `AAPL`, `MSFT`) or topics.
* **Dual Sentiment Analysis**:
    * **VADER**: Fast, rule-based sentiment analysis using a specific lexicon.
    * **FinBERT**: Deep, contextual analysis using the `ProsusAI/finbert` model from Transformers.
* **Text Cleaning**: Preprocesses news articles by removing URLs, stock tickers ($), and other artifacts.
* **Daily Aggregation**: Groups sentiment scores by day, generating features such as mean, standard deviation, and positive/negative news ratios.

## Requirements

Project dependencies are listed in `requirements.txt`. Key dependencies include:

* `pandas`
* `requests`
* `transformers`
* `torch`
* `nltk`
* `python-dotenv`

## Installation

 **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

To use the Alpha Vantage API, you need an API key.

1.  **Get your free API key** from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

2.  **Create your environment file:**
    Rename the `.env.example` file to `.env`.
    ```bash
    mv .env.example .env
    ```

3.  **Edit the `.env` file** and add your API key:
    ```ini
    ALPHA_VANTAGE_API_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

Once configured, you can run the main pipeline:

```bash
python main.py
