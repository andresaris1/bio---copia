from flask import Flask, render_template, request
from credit_risk_model import CreditRiskModel
import os
import psycopg2

app = Flask(__name__)
crm = CreditRiskModel()
crm.load()

# Crea el archivo credentials.json si no existe, usando la variable de entorno
if not os.path.exists('credentials.json') and 'GOOGLE_CREDENTIALS_JSON' in os.environ:
    with open('credentials.json', 'w') as f:
        f.write(os.environ['GOOGLE_CREDENTIALS_JSON'])

def guardar_resultado_en_db(score, probabilidad):
    conn = psycopg2.connect(
        host="dpg-d133a5emcj7s73fu4eh0-a.oregon-postgres.render.com",  # Host externo de Render
        database="datos_de_scorecard",
        user="datos_de_scorecard_user",
        password="yTZonId3v8h477Ch3rqIy8fmusl8C67r",
        port="5432"
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

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        datos_usuario = {
            'last_pymnt_amnt': float(request.form['last_pymnt_amnt']),
            'total_rec_prncp': float(request.form['total_rec_prncp']),
            'funded_amnt': float(request.form['funded_amnt']),
            'funded_amnt_inv': float(request.form['funded_amnt_inv']),
            'total_pymnt_inv': float(request.form['total_pymnt_inv']),
            'total_pymnt': float(request.form['total_pymnt'])
        }
        prob = crm.predict(datos_usuario)[0]
        score = int(crm.prob_to_score(prob))
        resultado = {
            'probabilidad': f"{prob*100:.2f}",
            'score': score
        }
        guardar_resultado_en_db(score, resultado['probabilidad'])
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)