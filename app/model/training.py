from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from main import logging



class ModelTrainer:
    def __init__(self, model_path='fraud_model.pkl'):
        self.model_path = Path(model_path)
        self.pipeline = None

    def train_model(self, df, scaler):
        logging.info("Training model.")
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', LogisticRegression(solver='liblinear'))
        ])
        self.pipeline.fit(X_train, y_train)
        
        y_pred = self.pipeline.predict(X_val)
        logging.info("Classification Report:\n" + classification_report(y_val, y_pred))
        
        joblib.dump(self.pipeline, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load_model(self):
        logging.info("Loading model.")
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        else:
            logging.error("Model file not found. Train the model first.")
            raise FileNotFoundError("Model file not found. Train the model first.")


class Predictor:
    def __init__(self, model_path='fraud_model.pkl'):
        self.model_path = Path(model_path)
        self.pipeline = None

    def load_model(self):
        logging.info("Loading model for prediction.")
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        else:
            logging.error("Model file not found.")
            raise FileNotFoundError("Model file not found. Train the model first.")

    def predict(self, new_data):
        logging.info("Predicting new data.")
        self.load_model()
        X_new = new_data.drop('Class', axis=1, errors='ignore')
        predictions = self.pipeline.predict(X_new)
        return predictions