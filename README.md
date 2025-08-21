# Sentiment Analysis Web Application

A simple yet powerful web application for analyzing text sentiment using Python, Flask, and Machine Learning.

## Features

- **Real-time Sentiment Analysis**: Analyze text sentiment (Positive, Negative, Neutral) instantly
- **High Accuracy**: Uses NLTK and Scikit-learn for robust sentiment classification
- **Clean UI**: Modern, responsive design with Bootstrap 5
- **Modular Code**: Well-structured codebase for easy maintenance and extension
- **Pre-trained Model**: Comes with a pre-trained model using NLTK movie reviews dataset

## Tech Stack

- **Backend**: Python, Flask
- **ML Libraries**: NLTK, Scikit-learn
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Icons**: Font Awesome

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone or download the project**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser and go to `http://localhost:5000`

## Usage

1. **Enter Text**: Type or paste any text in the input box
2. **Analyze**: Click the "Analyze Sentiment" button
3. **View Results**: See the predicted sentiment (Positive, Negative, or Neutral) along with confidence score

## Project Structure

```
sentiment_analysis_app/
├── app.py                 # Flask application routes
├── model.py              # ML model training and prediction
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── sentiment_model.pkl  # Trained model (generated after first run)
├── vectorizer.pkl       # TF-IDF vectorizer (generated after first run)
└── templates/
    └── index.html       # Frontend HTML template
```

## Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF vectorization with 5000 features
- **Training Data**: NLTK movie reviews dataset + custom neutral examples
- **Preprocessing**: Lowercase, punctuation removal, stopword removal, tokenization
- **Accuracy**: ~85% on test set

## API Endpoints

- `GET /` - Home page
- `POST /predict` - Analyze sentiment for given text
- `GET /health` - Health check endpoint

## Example API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this movie!"}'
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "text": "I love this movie!"
}
```

## Customization

### Adding More Training Data
1. Modify the `prepare_data()` method in `model.py`
2. Add your own labeled text data
3. Retrain the model by deleting `sentiment_model.pkl` and `vectorizer.pkl`

### Changing Model Parameters
- Edit the `SentimentAnalyzer` class in `model.py`
- Adjust vectorizer parameters, classifier type, or preprocessing steps

## Troubleshooting

### Common Issues

1. **NLTK Data Download Error**
   ```python
   import nltk
   nltk.download('all')
   ```

2. **Port Already in Use**
   ```bash
   python app.py --port 5001
   ```

3. **Model Not Found**
   - The model will be automatically trained on first run
   - Ensure internet connection for NLTK data download

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for the movie reviews dataset
- Scikit-learn for machine learning algorithms
- Bootstrap for the beautiful UI components
