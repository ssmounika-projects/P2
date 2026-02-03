from flask import Flask, render_template, request, jsonify
import re
import os
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
from scipy.sparse import load_npz

app = Flask(__name__, 
    template_folder=os.path.join(os.getcwd(), 'templates'),
    static_folder=os.path.join(os.getcwd(), 'static'),
    static_url_path='/static'
)

# Load the trained model and vectorizer
try:
    model = joblib.load('best_model.pkl')
    with open('unigram_bow.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Text preprocessing function
def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return text

# Route for the home page
@app.route('/')
def index():
    return render_template('frontpage.html')

# Route to validate SQL query
@app.route('/validate', methods=['POST'])
def validate_query():
    try:
        query = request.json.get('query')
        result = {}
        
        # Preprocess the query
        processed_query = preprocess_text(query)
        
        # Transform the query using the vectorizer
        query_vec = vectorizer.transform([processed_query])
        
        # Make prediction
        prediction = model.predict(query_vec)[0]
        probability = model.predict_proba(query_vec)[0][1]  # Probability of being malicious
        
        result['safe'] = bool(prediction == 0)  # 0 for benign, 1 for malicious
        result['message'] = 'SQL Query is Safe.' if result['safe'] else 'Warning: Potential SQL Injection Detected!'
        result['dl_score'] = float(1 - probability)  # Convert to float for JSON serialization
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Download required NLTK data
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
    
    app.run(debug=True)
