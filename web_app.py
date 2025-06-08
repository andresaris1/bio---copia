from flask import Flask, render_template, request
from credit_risk_model import CreditRiskModel
import gspread
from google.oauth2.service_account import Credentials
import os

app = Flask(__name__)
crm = CreditRiskModel()
crm.load()

# Crea el archivo credentials.json si no existe, usando la variable de entorno
if not os.path.exists('credentials.json') and 'GOOGLE_CREDENTIALS_JSON' in os.environ:
    with open('credentials.json', 'w') as f:
        f.write(os.environ['GOOGLE_CREDENTIALS_JSON'])

def guardar_resultado_en_sheets(score, probabilidad):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open('Resultados de la encuesta')  # Cambiado al nombre correcto
    worksheet = sh.sheet1  # O usa .worksheet('NombreDeLaHoja') si tienes varias
    fila = [score, probabilidad]
    worksheet.append_row(fila)

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
        guardar_resultado_en_sheets(score, resultado['probabilidad'])
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)