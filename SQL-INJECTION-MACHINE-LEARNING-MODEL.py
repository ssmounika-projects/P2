import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
import nltk
import ssl

class QueryAnalyzer:
    @staticmethod
    def setup_nltk():
        """Initialize NLTK with SSL handling"""
        try:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("NLTK data downloaded successfully")
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")
            raise

    @staticmethod
    def clean_text(text):
        """Process and clean input text"""
        try:
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = nltk.word_tokenize(text)
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            stemmer = nltk.PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return text

    @staticmethod
    def process_dataframe(df):
        """Process DataFrame containing queries"""
        try:
            df_processed = df.copy()
            df_processed['Query'] = df_processed['Query'].apply(QueryAnalyzer.clean_text)
            return df_processed
        except Exception as e:
            print(f"Error in DataFrame preprocessing: {str(e)}")
            return df

class FeatureExtractor:
    @staticmethod
    def calculate_query_metrics(text):
        """Extract 6 key metrics for SQL injection detection"""
        if not isinstance(text, str) or not text.strip():
            return np.zeros(6)
        
        text = str(text).strip()
        total_chars = max(len(text), 1)
        
        # 1. Query Length Metric
        length_metric = len(text) / 1000
        
        # 2. Special Character Density
        special_chars = set("=,;'\"--/*()[]{}<>!+-*/%")
        special_char_count = sum(1 for c in text if c in special_chars)
        special_char_ratio = special_char_count / total_chars
        
        # 3. Pattern Matching Score
        patterns = [
            r'\d+=\d+',
            r'[a-zA-Z]+=[a-zA-Z]+',
            r'[a-zA-Z]+\s*=\s*[a-zA-Z]+',
            r'\d+\s*=\s*\d+'
        ]
        pattern_count = sum(len(re.findall(pattern, text)) for pattern in patterns)
        pattern_ratio = pattern_count / total_chars
        
        # 4. Character Diversity Score
        alpha_count = sum(1 for c in text if c.isalpha())
        num_count = sum(1 for c in text if c.isdigit())
        symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        diversity_score = (alpha_count * num_count * symbol_count) / (total_chars ** 3) if total_chars > 0 else 0
        
        # 5. SQL Command Frequency
        sql_commands = [
            'SELECT', 'UNION', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER',
            'EXEC', 'EXECUTE', 'DECLARE', 'CAST', 'CONVERT', 'UNHEX', 'HEX',
            'CHAR', 'ASCII', 'SUBSTRING', 'CONCAT'
        ]
        command_count = sum(text.upper().count(cmd) for cmd in sql_commands)
        command_ratio = command_count / total_chars
        
        # 6. Structural Complexity
        space_count = sum(1 for c in text if c.isspace())
        comment_count = text.count('--') + text.count('/*') + text.count('*/')
        control_count = sum(1 for c in text if c in '\n\r\t')
        structure_ratio = (space_count + comment_count + control_count) / total_chars
        
        return np.array([
            length_metric,
            special_char_ratio,
            pattern_ratio,
            diversity_score,
            command_ratio,
            structure_ratio
        ])

    @staticmethod
    def extract_features(X_train, X_test):
        """Extract features from training and test data"""
        try:
            # N-gram Features
            ngram_vectorizer = CountVectorizer(
                ngram_range=(1,2),
                min_df=5,
                max_df=0.95,
                max_features=2000,
                token_pattern=r'(?u)\b\w\w+\b',
                strip_accents='unicode',
                lowercase=True,
                stop_words='english'
            )
            X_train_ngram = ngram_vectorizer.fit_transform(X_train)
            X_test_ngram = ngram_vectorizer.transform(X_test)
            
            # TF-IDF Features
            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1,2),
                min_df=5,
                max_df=0.95,
                max_features=2000,
                token_pattern=r'(?u)\b\w\w+\b',
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)
            
            # Feature Metrics
            X_train_metrics = np.array([FeatureExtractor.calculate_query_metrics(text) for text in X_train])
            X_test_metrics = np.array([FeatureExtractor.calculate_query_metrics(text) for text in X_test])
            
            # Handle NaN and infinite values
            X_train_metrics = np.nan_to_num(X_train_metrics, nan=0.0, posinf=1.0, neginf=0.0)
            X_test_metrics = np.nan_to_num(X_test_metrics, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Apply log transformation
            X_train_metrics = np.log1p(np.abs(X_train_metrics))
            X_test_metrics = np.log1p(np.abs(X_test_metrics))
            
            # Scale features
            scaler = RobustScaler()
            X_train_metrics = scaler.fit_transform(X_train_metrics)
            X_test_metrics = scaler.transform(X_test_metrics)
            
            return {
                'ngram': {'train': X_train_ngram, 'test': X_test_ngram},
                'tfidf': {'train': X_train_tfidf, 'test': X_test_tfidf},
                'feature_metrics': {'train': X_train_metrics, 'test': X_test_metrics}
            }
        
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

class ModelEvaluator:
    @staticmethod
    def train_model(model, X_train, X_test, y_train, y_test, model_name, feature_type):
        """Train and evaluate a model"""
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
            }
            
            return metrics
        
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

    @staticmethod
    def train_naive_bayes(X_train, X_test, y_train, y_test, feature_type):
        """Train and evaluate Gaussian Naive Bayes model"""
        try:
            if feature_type == 'feature_metrics':
                # Convert to numpy arrays if not already
                X_train = np.array(X_train)
                X_test = np.array(X_test)
                
                # Handle NaN and infinite values
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Apply log transformation
                X_train = np.log1p(np.abs(X_train))
                X_test = np.log1p(np.abs(X_test))
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Feature selection
                selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=200, random_state=42),
                    max_features=6
                )
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)
                
                model = GaussianNB()
                
            else:
                # For n-gram and TF-IDF features
                X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model = GaussianNB()
        
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
            }
            
            return metrics
        
        except Exception as e:
            print(f"Error in Naive Bayes training: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

class ResultVisualizer:
    @staticmethod
    def create_performance_table(results, feature_types):
        """Create a formatted table of model performance"""
        table = PrettyTable()
        table.field_names = ["Model", "Feature Type", "Accuracy", "Precision", "Recall", "F1 Score"]
        table.hrules = True
        table.vrules = True
        
        feature_type_map = {
            'ngram': 'N-gram',
            'tfidf': 'TF-IDF',
            'feature_metrics': 'Feature Metrics'
        }
        
        for model_name in sorted(results.keys()):
            for feature_type in sorted(feature_types):
                metrics = results[model_name][feature_type]
                table.add_row([
                    model_name,
                    feature_type_map[feature_type],
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}"
                ])
        
        return table
        
    @staticmethod
    def plot_model_comparison(results, feature_types):
        """Plot model comparison charts"""
        plt.figure(figsize=(15, 5))
        
        model_names = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.25
        
        metrics = ['accuracy', 'f1']
        titles = ['Accuracy', 'F1 Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 2, i+1)
            for j, feature_type in enumerate(feature_types):
                values = [results[model][feature_type][metric] for model in model_names]
                plt.bar(x + j*width, values, width, 
                       label=feature_type.replace('_', ' ').title())
            
            plt.title(f'Model {title} Comparison')
            plt.xlabel('Models')
            plt.ylabel(title)
            plt.xticks(x + width, model_names, rotation=45)
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    try:
        # Setup NLTK
        QueryAnalyzer.setup_nltk()
        
        # Load and preprocess data
        if not os.path.exists('queries.csv'):
            raise FileNotFoundError("queries.csv not found")
        
        data = pd.read_csv('queries.csv')
        data = QueryAnalyzer.process_dataframe(data)
        
        # Split data
        X = data['Query']
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Extract features
        features = FeatureExtractor.extract_features(X_train, X_test)
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=500, n_jobs=-1, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'Naive Bayes': None,
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        }
        
        # Train and evaluate models
        results = {}
        
        for model_name, model in models.items():
            if model_name != 'Naive Bayes':
                results[model_name] = {}
                for feature_type, feature_data in features.items():
                    metrics = ModelEvaluator.train_model(
                        model,
                        feature_data['train'],
                        feature_data['test'],
                        y_train,
                        y_test,
                        model_name,
                        feature_type
                    )
                    results[model_name][feature_type] = metrics
        
        # Train Naive Bayes
        results['Naive Bayes'] = {}
        for feature_type, feature_data in features.items():
            metrics = ModelEvaluator.train_naive_bayes(
                feature_data['train'],
                feature_data['test'],
                y_train,
                y_test,
                feature_type
            )
            results['Naive Bayes'][feature_type] = metrics
        
        # Visualize results
        print("\nModel Performance Summary:")
        table = ResultVisualizer.create_performance_table(results, features.keys())
        print(table.get_string())
        
        ResultVisualizer.plot_model_comparison(results, features.keys())
        
        print("\nExecution completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
