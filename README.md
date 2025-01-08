# Using Natural Language Processing on News Data to Predict Share Prices

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Kareemcanaan/sentiment-analysis)

## Project Overview
This project explores the use of **Natural Language Processing (NLP)** techniques for **financial sentiment analysis** to predict stock price movements. The goal is to automate the analysis a trader might perform when reading financial news articles and to determine whether a stock’s price will increase or decrease over the next 24 hours.

The models implemented include:
1. **Naive Bayes Classifier**
2. **Long Short-Term Memory (LSTM) Networks**
3. **Pre-trained FinBERT**
4. **Fine-Tuned BERT**

## Key Features
- **Live Data Collection**: The project uses **NewsAPI** and **YFinance** to gather up-to-date financial news and stock price data.
- **Data Preprocessing**: 
  - Tokenization
  - Stopword removal
  - Lemmatization
  - Cleaning to remove HTML tags and non-alphanumeric characters
- **Sentiment Analysis Models**:
  - Naive Bayes Classifier: Uses Bag of Words with Bayesian probability.
  - LSTM: Built using Keras and TensorFlow libraries.
  - FinBERT: Pre-trained model via Hugging Face API.
  - Fine-tuned BERT: DistilBERT model fine-tuned using custom datasets.
- **Performance Metrics**: Accuracy, F1 Score, confusion matrix, and returns were used to evaluate the models.
  
## Results
- The **Naive Bayes** and **LSTM** models consistently outperformed FinBERT and fine-tuned BERT in terms of accuracy and F1 score.
- **Naive Bayes** was particularly effective for smaller datasets due to its simplicity and lower computational requirements.
- **Portfolio Construction**: A zero-investment portfolio was constructed based on the sentiment analysis results, showing promising potential returns.

## Limitations
- **Data Availability**: Limited by the 30-day lookback period of the free tier of NewsAPI.
- **Trading Fees**: Ignored in return calculations.
- **Fine-tuning BERT**: Limited performance due to using DistilBERT and small dataset size.
  
## Future Work
- Obtain more extensive datasets for better training and evaluation.
- Explore other pre-trained models, including GPT-based models.
- Improve data preprocessing by addressing edge cases like HTML tags.
- Include trading fees and slippage in return calculations for more realistic performance.

## Requirements
- Python 3.x
- TensorFlow / Keras
- Scikit-Learn
- Hugging Face Transformers
- NLTK
- NewsAPI and YFinance libraries

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Kareemcanaan/sentiment-analysis.git
   cd sentiment-analysis
2. Install the required packages:
  ```bash
pip install -r requirements.txt
```
## Acknowledgements
This project was based on the dissertation project of Nick Bispham
Special thanks to Dr. Tingting Mu for guidance and Dr. Eghbal Rahimikia for assistance with finance-related queries.
