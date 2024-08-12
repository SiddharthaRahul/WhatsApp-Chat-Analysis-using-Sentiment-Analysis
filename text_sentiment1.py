import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
import numpy as np

# Function to load BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = TFBertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

# Function to analyze sentiment in batches
def analyze_sentiment_batch(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    outputs = model(inputs)
    predictions = np.argmax(outputs.logits, axis=1)
    return predictions

# Example Streamlit app code
def get_score(user, df):
    st.title("Sentiment Analysis with BERT")
    df = df[df['user'] == user] if user != 'Overall' else df

        # Load model and tokenizer
    tokenizer, model = load_model()

    # Process data in batches
    batch_size = 32
    num_batches = len(df) // batch_size
    sentiment_predictions = []

    for batch_idx in range(num_batches + 1):
        batch_texts = df['message'][batch_idx * batch_size:(batch_idx + 1) * batch_size].tolist()
        if batch_texts:
            batch_predictions = analyze_sentiment_batch(batch_texts, tokenizer, model)
            sentiment_predictions.extend(batch_predictions)

    if len(sentiment_predictions) < len(df):
        sentiment_predictions.extend([np.nan] * (len(df) - len(sentiment_predictions)))
    
    df['sentiment'] = sentiment_predictions

    st.dataframe(df)

