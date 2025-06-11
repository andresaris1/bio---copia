import os
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras import layers

class CreditRiskModel:
    # Variables más importantes según KBest
    TOP_FEATURES = [
        'out_prncp',
        'out_prncp_inv',
        'total_pymnt',
        'total_pymnt_inv',
        'total_rec_prncp',
        'recoveries',
        'last_pymnt_amnt'
    ]

    def __init__(self, model_path='credit_risk_model.h5', prep_path='preprocessing.pkl', scaler_path='scaler.joblib'):
        self.model_path = model_path
        self.prep_path = prep_path
        self.scaler_path = scaler_path
        self.model = None
        self.columns = None
        self.scaler = None

    def _clean_dataframe(self, df):
        # Elimina columnas con muchos nulos y no significativas
        columns_to_remove = [
            'earliest_cr_line', 'last_credit_pull_d', 'last_pymnt_d', 'emp_length',
            'next_pymnt_d', 'verification_status_joint', 'desc', 'emp_title', 'title'
        ]
        df = df.drop(columns=columns_to_remove, errors='ignore')
        # Elimina columnas problemáticas
        cols_to_drop = ['id', 'member_id', 'url', 'zip_code', 'loan_status']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        # Elimina columnas con un solo valor
        df = df.drop(columns=["application_type"], errors='ignore')
        # One-hot encoding a las variables categóricas
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        # Imputa nulos
        df = df.fillna(df.median(numeric_only=True))
        return df

    def preprocess(self, df):
        df_clean = self._clean_dataframe(df)
        for col in self.TOP_FEATURES:
            if col not in df_clean.columns:
                df_clean[col] = 0
        df_clean = df_clean[self.TOP_FEATURES]
        if self.scaler is not None:
            df_clean = pd.DataFrame(self.scaler.transform(df_clean), columns=self.TOP_FEATURES)
        return df_clean

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
        df_clean = self._clean_dataframe(df)
        X = df_clean[self.TOP_FEATURES]
        y = df['target']
        # Escalado
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        X = X.astype('float32')
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.TOP_FEATURES, self.prep_path)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        # Red neuronal igual que en el notebook
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        model.save(self.model_path)
        self.model = model
        # Evaluación
        from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
        y_pred_prob = model.predict(X_test)
        y_pred_classes = (y_pred_prob > 0.5).astype(int)
        print("\nNeural Network Performance Metrics:")
        print("--------------------------------")
        print(f"AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        cm = confusion_matrix(y_test, y_pred_classes)
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Neural Network Confusion Matrix')
        plt.show()

    def load(self):
        self.model = keras.models.load_model(self.model_path)
        self.columns = joblib.load(self.prep_path)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler = None

    def predict(self, input_data):
        if self.model is None or self.scaler is None:
            self.load()
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        df_proc = self.preprocess(df)
        preds = self.model.predict(df_proc).ravel()
        return preds

    def prob_to_score(self, prob, min_score=300, max_score=850):
        return min_score + (max_score - min_score) * (1 - prob)

# Entrenamiento desde cero
if __name__ == '__main__':
    model_file = 'credit_risk_model.h5'
    prep_file = 'preprocessing.pkl'
    scaler_file = 'scaler.joblib'
    if os.path.exists(model_file):
        os.remove(model_file)
    if os.path.exists(prep_file):
        os.remove(prep_file)
    if os.path.exists(scaler_file):
        os.remove(scaler_file)
    crm = CreditRiskModel(model_file, prep_file, scaler_file)
    crm.fit(os.path.join('loan', 'loan.csv'))
    print('Modelo entrenado y guardado.')
    # Ejemplo de predicción
    ejemplo = {
        'last_pymnt_amnt': 500,
        'total_rec_prncp': 10000,
        'out_prncp': 0,
        'out_prncp_inv': 0,
        'total_pymnt_inv': 11000,
        'total_pymnt': 11000,
        'recoveries': 0
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
            'out_prncp': 0,
            'out_prncp_inv': 0,
            'total_pymnt_inv': 13000,
            'total_pymnt': 13000,
            'recoveries': 0
        },
        # Cliente con saldo pendiente alto y pocos pagos (alto riesgo)
        {
            'last_pymnt_amnt': 0,
            'total_rec_prncp': 1000,
            'out_prncp': 9000,
            'out_prncp_inv': 9000,
            'total_pymnt_inv': 1200,
            'total_pymnt': 1200,
            'recoveries': 0
        },
        # Cliente con pagos recientes medianos y saldo medio
        {
            'last_pymnt_amnt': 500,
            'total_rec_prncp': 6000,
            'out_prncp': 4000,
            'out_prncp_inv': 4000,
            'total_pymnt_inv': 7000,
            'total_pymnt': 7000,
            'recoveries': 0
        },
        # Cliente que ya pagó todo (cero saldo, pagos completos)
        {
            'last_pymnt_amnt': 0,
            'total_rec_prncp': 15000,
            'out_prncp': 0,
            'out_prncp_inv': 0,
            'total_pymnt_inv': 16000,
            'total_pymnt': 16000,
            'recoveries': 0
        },
        # Cliente con pagos recientes altos pero saldo pendiente
        {
            'last_pymnt_amnt': 2000,
            'total_rec_prncp': 5000,
            'out_prncp': 5000,
            'out_prncp_inv': 5000,
            'total_pymnt_inv': 7000,
            'total_pymnt': 7000,
            'recoveries': 0
        }
    ]
    print("\n--- Ejemplos adicionales de predicción ---")
    for i, ej in enumerate(ejemplos, 1):
        prob = crm.predict(ej)[0]
        score = crm.prob_to_score(prob)
        print(f"Ejemplo {i}: {ej}")
        print(f"  Probabilidad de incumplimiento: {prob:.2%}")
        print(f"  Score crediticio: {score:.0f}\n")