import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler else RobustScaler()

    def read_data(self, file_path):
        logging.info("Reading data from file.")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error reading file: {e}")
            raise

    def scale_features(self, df):
        logging.info("Scaling features.")
        df['scaled_amount'] = self.scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = self.scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        return df

    def preprocess_new_data(self, new_data):
        logging.info("Preprocessing new data.")
        new_data['scaled_amount'] = self.scaler.transform(new_data['Amount'].values.reshape(-1, 1))
        new_data['scaled_time'] = self.scaler.transform(new_data['Time'].values.reshape(-1, 1))
        new_data.drop(['Time', 'Amount'], axis=1, inplace=True)
        return new_data

    def balance_classes(self, df):
        logging.info("Balancing classes.")
        df = df.sample(frac=1, random_state=42)
        fraud_df = df.loc[df['Class'] == 1]
        non_fraud_df = df.loc[df['Class'] == 0][:492]
        normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
        return normal_distributed_df.sample(frac=1, random_state=42)

    def log_stats(self, df):
        logging.info(f'No Frauds: {round(df["Class"].value_counts()[0] / len(df) * 100, 2)}% of the dataset')
        logging.info(f'Frauds: {round(df["Class"].value_counts()[1] / len(df) * 100, 2)}% of the dataset')



# # Usage example
# file_path = 'creditcard.csv'
# preprocessor = Preprocessor()
# df = preprocessor.read_data(file_path)
# preprocessor.log_stats(df)
# scaled_df = preprocessor.scale_features(df)
# balanced_df = preprocessor.balance_classes(scaled_df)

# model_trainer = ModelTrainer()
# model_trainer.train_model(balanced_df, preprocessor.scaler)

# new_data_path = 'new_creditcard.csv'
# new_data = preprocessor.read_data(new_data_path)
# new_data_processed = preprocessor.preprocess_new_data(new_data)

# predictor = Predictor()
# predictions = predictor.predict(new_data_processed)
# logging.info(f'Predictions: {predictions}')
