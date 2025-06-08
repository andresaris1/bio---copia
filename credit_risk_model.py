import os
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras import layers

class CreditRiskModel:
    def __init__(self, model_path='credit_risk_model.h5', prep_path='preprocessing.pkl'):
        self.model_path = model_path
        self.prep_path = prep_path
        self.model = None
        self.columns = None

    def preprocess(self, df):
        # Elimina columnas problemáticas
        cols_to_drop = [
            'id', 'member_id', 'url', 'desc', 'title', 'zip_code', 'loan_status', 'emp_title'
        ]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        # Llena nulos simples
        df = df.fillna(df.median(numeric_only=True))
        # Convierte categóricas a dummies
        df = pd.get_dummies(df, drop_first=True)
        # Asegura el mismo orden de columnas
        if self.columns is not None:
            for col in self.columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.columns]
        return df

    def fit(self, csv_path):
        df = pd.read_csv(csv_path)
        # Mapeo de variable objetivo
        status_map = {
            "Fully Paid": 0,
            "Charged Off": 1,
            "Late (31-120 days)": 1,
            "Default": 1,
            "Does not meet the credit policy. Status:Fully Paid": 0,
            "Does not meet the credit policy. Status:Charged Off": 1
        }
        df['target'] = df['loan_status'].map(status_map)
        df_model = df[df['target'].notna()].copy()
        df_model = self.preprocess(df_model)
        X = df_model.drop('target', axis=1)
        y = df_model['target']
        self.columns = X.columns.tolist()
        # Guarda columnas para el preprocesamiento
        joblib.dump(self.columns, self.prep_path)
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Modelo
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1)
        model.save(self.model_path)
        self.model = model

    def load(self):
        self.model = keras.models.load_model(self.model_path)
        self.columns = joblib.load(self.prep_path)

    def predict(self, input_data):
        if self.model is None or self.columns is None:
            self.load()
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        df_proc = self.preprocess(df)
        preds = self.model.predict(df_proc)
        return preds.ravel()

    def prob_to_score(self, prob, min_score=300, max_score=850):
        return min_score + (max_score - min_score) * (1 - prob)

# Ejemplo de entrenamiento (solo la primera vez):
if __name__ == '__main__':
    model_file = 'credit_risk_model.h5'
    prep_file = 'preprocessing.pkl'
    if not os.path.exists(model_file) or not os.path.exists(prep_file):
        crm = CreditRiskModel(model_file, prep_file)
        crm.fit(os.path.join('loan', 'loan.csv'))
        print('Modelo entrenado y guardado.')
    else:
        print('Modelo ya entrenado. Usa la clase para predecir.') 