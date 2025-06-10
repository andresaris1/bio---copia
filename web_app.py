from flask import Flask, render_template, request
from credit_risk_model import CreditRiskModel
import os
import psycopg2
import json
import numpy as np

app = Flask(__name__)
crm = CreditRiskModel()
crm.load()

# Crea el archivo credentials.json si no existe, usando la variable de entorno
if not os.path.exists('credentials.json') and 'GOOGLE_CREDENTIALS_JSON' in os.environ:
    with open('credentials.json', 'w') as f:
        f.write(os.environ['GOOGLE_CREDENTIALS_JSON'])

def guardar_resultado_en_db(score, probabilidad):
    conn = psycopg2.connect(
        host=os.environ['DB_HOST'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        port=os.environ.get('DB_PORT', 5432)
    )
    cur = conn.cursor()
    # Crear la tabla si no existe
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resultados (
        id SERIAL PRIMARY KEY,
        score INTEGER,
        probabilidad FLOAT,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    # Insertar el resultado
    cur.execute(
        "INSERT INTO resultados (score, probabilidad) VALUES (%s, %s)",
        (score, float(probabilidad))
    )
    conn.commit()
    cur.close()
    conn.close()

def obtener_scores_poblacion():
    conn = psycopg2.connect(
        host=os.environ['DB_HOST'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        port=os.environ.get('DB_PORT', 5432)
    )
    cur = conn.cursor()
    cur.execute("SELECT score FROM resultados")
    scores = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return scores

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    percentiles = None
    user_percentil = None
    if request.method == 'POST':
        datos_usuario = {
            'last_pymnt_amnt': float(request.form['last_pymnt_amnt']),
            'total_rec_prncp': float(request.form['total_rec_prncp']),
            'funded_amnt': float(request.form['funded_amnt']),
            'funded_amnt_inv': float(request.form['funded_amnt_inv']),
            'total_pymnt_inv': float(request.form['total_pymnt_inv']),
            'total_pymnt': float(request.form['total_pymnt']),
            'recoveries': float(request.form['recoveries'])
        }
        prob = crm.predict(datos_usuario)[0]
        score = int(crm.prob_to_score(prob))
        # Clasificación del score
        if score > 800:
            clasificacion = "Buen crédito"
        elif 400 <= score <= 800:
            clasificacion = "Crédito regular"
        else:
            clasificacion = "Mal crédito"
        # Obtener scores de la población y calcular percentiles
        scores_poblacion = obtener_scores_poblacion()
        if scores_poblacion:
            percentiles = [np.percentile(scores_poblacion, p) for p in [0, 25, 50, 75, 100]]
            user_percentil = int((np.sum(np.array(scores_poblacion) <= score) / len(scores_poblacion)) * 100)
        resultado = {
            'probabilidad': f"{prob*100:.2f}",
            'score': score,
            'clasificacion': clasificacion,
            'user_percentil': user_percentil
        }
        guardar_resultado_en_db(score, resultado['probabilidad'])
    return render_template('index.html', resultado=resultado, percentiles=percentiles)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)