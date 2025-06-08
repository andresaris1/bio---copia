import joblib

columnas = joblib.load('preprocessing.pkl')
for col in columnas:
    print(col) 