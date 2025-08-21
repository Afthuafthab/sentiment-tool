import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.model_trained = False
        self.model_path = 'sentiment_model.pkl'
        self.vectorizer_path = 'vectorizer.pkl'
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load or train model
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.load_model()
        else:
            self.train_model()
            self.save_model()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/movie_reviews')
        except LookupError:
            nltk.download('movie_reviews')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Join tokens back
        return ' '.join(tokens)
    
    def prepare_data(self):
        """Prepare training data from NLTK movie reviews"""
        documents = []
        labels = []
        
        # Load positive reviews
        for fileid in movie_reviews.fileids('pos'):
            documents.append(movie_reviews.raw(fileid))
            labels.append('positive')
        
        # Load negative reviews
        for fileid in movie_reviews.fileids('neg'):
            documents.append(movie_reviews.raw(fileid))
            labels.append('negative')
        
        # Add some neutral examples (using simple heuristics)
        neutral_texts = [
            "This movie is okay, nothing special.",
            "It's fine, not great but not bad either.",
            "Average film with some good and bad parts.",
            "Mediocre performance overall.",
            "Neither impressive nor disappointing.",
            "The movie was alright, I guess.",
            "Standard fare, nothing remarkable.",
            "It was okay, I have mixed feelings.",
            "Not terrible, but not amazing either.",
            "Just another average movie."
        ]
        
        for text in neutral_texts:
            documents.append(text)
            labels.append('neutral')
        
        return documents, labels
    
    def train_model(self):
        """Train the sentiment analysis model"""
        print("Training sentiment analysis model...")
        
        # Prepare data
        documents, labels = self.prepare_data()
        
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_docs, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        self.model_trained = True
    
    def predict_sentiment(self, text):
        """Predict sentiment for given text"""
        if not self.model_trained:
            raise Exception("Model not trained")
        
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(vectorized_text)[0]
        
        return prediction
    
    def get_confidence(self, text):
        """Get prediction confidence"""
        if not self.model_trained:
            return 0.0
        
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        probabilities = self.classifier.predict_proba(vectorized_text)[0]
        confidence = max(probabilities)
        
        return float(confidence)
    
    def save_model(self):
        """Save trained model and vectorizer"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_model(self):
        """Load pre-trained model and vectorizer"""
        with open(self.model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.model_trained = True

# For testing purposes
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test predictions
    test_texts = [
        "I love this movie! It's amazing!",
        "This is the worst film I've ever seen.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        sentiment = analyzer.predict_sentiment(text)
        confidence = analyzer.get_confidence(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
        print("-" * 50)
