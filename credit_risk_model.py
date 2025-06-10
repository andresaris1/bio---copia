import os
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras import layers

class CreditRiskModel:
    # Solo las 7 variables más importantes según SHAP/KBest
    TOP_FEATURES = [
        'last_pymnt_amnt',
        'total_rec_prncp',
        'funded_amnt',
        'funded_amnt_inv',
        'total_pymnt_inv',
        'total_pymnt',
        'recoveries'
    ]

    def __init__(self, model_path='credit_risk_model.h5', prep_path='preprocessing.pkl'):
        self.model_path = model_path
        self.prep_path = prep_path
        self.model = None
        self.columns = None

    def preprocess(self, df):
        df = df.fillna(df.median(numeric_only=True))
        for col in self.TOP_FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[self.TOP_FEATURES]
        self.columns = self.TOP_FEATURES
        df = df.astype('float32')
        return df

    def fit(self, csv_path):
        df = pd.read_csv(csv_path)
        status_map = {
            "Fully Paid": 0,
            "Charged Off": 1,
            "Late (31-120 days)": 1,
            "Default": 1,
            "Does not meet the credit policy. Status:Fully Paid": 0,
            "Does not meet the credit policy. Status:Charged Off": 1
        }
        df['target'] = df['loan_status'].map(status_map)
        df = df[df['target'].notna()].copy()
        y = df['target']  # Separa el target antes del preprocesamiento
        X = df[self.TOP_FEATURES]
        X = self.preprocess(X)
        joblib.dump(self.columns, self.prep_path)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1)
        model.save(self.model_path)
        self.model = model
        # Evaluación
        from sklearn.metrics import roc_auc_score, classification_report
        y_pred_prob = model.predict(X_test).ravel()
        print("\n--- Evaluación en test ---")
        print("AUC:", roc_auc_score(y_test, y_pred_prob))
        print(classification_report(y_test, y_pred_prob > 0.5))

        # Calibración de probabilidades (Platt Scaling)
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()
        calibrator.fit(y_pred_prob.reshape(-1, 1), y_test)
        y_pred_prob_calibrated = calibrator.predict_proba(y_pred_prob.reshape(-1, 1))[:, 1]
        print("AUC calibrado:", roc_auc_score(y_test, y_pred_prob_calibrated))
        # Visualiza la distribución de scores calibrados
        def prob_to_score(prob, min_score=300, max_score=850):
            return min_score + (max_score - min_score) * (1 - prob)
        scores_calibrated = prob_to_score(y_pred_prob_calibrated)
        import matplotlib.pyplot as plt
        plt.hist(scores_calibrated, bins=30, edgecolor='k')
        plt.title('Distribución de Score Crediticio (Calibrado)')
        plt.xlabel('Score')
        plt.ylabel('Frecuencia')
        plt.show()
        # Guarda el calibrador para usarlo en predicción
        joblib.dump(calibrator, 'calibrator.pkl')
        self.calibrator = calibrator

    def load(self):
        self.model = keras.models.load_model(self.model_path)
        self.columns = joblib.load(self.prep_path)
        if os.path.exists('calibrator.pkl'):
            self.calibrator = joblib.load('calibrator.pkl')
        else:
            self.calibrator = None

    def predict(self, input_data, calibrate=True):
        if self.model is None or self.columns is None:
            self.load()
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        df_proc = self.preprocess(df)
        preds = self.model.predict(df_proc).ravel()
        if calibrate and hasattr(self, 'calibrator') and self.calibrator is not None:
            preds = self.calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
        return preds

    def prob_to_score(self, prob, min_score=300, max_score=850):
        return min_score + (max_score - min_score) * (1 - prob)

# Entrenamiento desde cero
if __name__ == '__main__':
    model_file = 'credit_risk_model.h5'
    prep_file = 'preprocessing.pkl'
    if os.path.exists(model_file):
        os.remove(model_file)
    if os.path.exists(prep_file):
        os.remove(prep_file)
    crm = CreditRiskModel(model_file, prep_file)
    crm.fit(os.path.join('loan', 'loan.csv'))
    print('Modelo entrenado y guardado.')
    # Ejemplo de predicción
    ejemplo = {
        'last_pymnt_amnt': 500,
        'total_rec_prncp': 10000,
        'funded_amnt': 12000,
        'funded_amnt_inv': 12000,
        'total_pymnt_inv': 11000,
        'total_pymnt': 11000
    }
    prob = crm.predict(ejemplo)[0]
    score = crm.prob_to_score(prob)
    print(f"Probabilidad de incumplimiento: {prob:.2%}")
    print(f"Score crediticio: {score:.0f}")

    # Ejemplos adicionales para corroborar el modelo
    ejemplos = [
        # Cliente con pagos altos y saldo bajo (bajo riesgo)
        {
            'last_pymnt_amnt': 100,
            'total_rec_prncp': 12000,
            'funded_amnt': 12000,
            'funded_amnt_inv': 12000,
            'total_pymnt_inv': 13000,
            'total_pymnt': 13000
        },
        # Cliente con saldo pendiente alto y pocos pagos (alto riesgo)
        {
            'last_pymnt_amnt': 0,
            'total_rec_prncp': 1000,
            'funded_amnt': 10000,
            'funded_amnt_inv': 10000,
            'total_pymnt_inv': 1200,
            'total_pymnt': 1200
        },
        # Cliente con pagos recientes medianos y saldo medio
        {
            'last_pymnt_amnt': 500,
            'total_rec_prncp': 6000,
            'funded_amnt': 10000,
            'funded_amnt_inv': 10000,
            'total_pymnt_inv': 7000,
            'total_pymnt': 7000
        },
        # Cliente que ya pagó todo (cero saldo, pagos completos)
        {
            'last_pymnt_amnt': 0,
            'total_rec_prncp': 15000,
            'funded_amnt': 15000,
            'funded_amnt_inv': 15000,
            'total_pymnt_inv': 16000,
            'total_pymnt': 16000
        },
        # Cliente con pagos recientes altos pero saldo pendiente
        {
            'last_pymnt_amnt': 2000,
            'total_rec_prncp': 5000,
            'funded_amnt': 10000,
            'funded_amnt_inv': 10000,
            'total_pymnt_inv': 7000,
            'total_pymnt': 7000
        }
    ]
    print("\n--- Ejemplos adicionales de predicción (calibrados) ---")
    for i, ej in enumerate(ejemplos, 1):
        prob = crm.predict(ej)[0]
        score = crm.prob_to_score(prob)
        print(f"Ejemplo {i}: {ej}")
        print(f"  Probabilidad de incumplimiento (calibrada): {prob:.2%}")
        print(f"  Score crediticio (calibrado): {score:.0f}\n") 