from flask import Flask, render_template, request
from credit_risk_model import CreditRiskModel

app = Flask(__name__)
crm = CreditRiskModel()
crm.load()

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        datos_usuario = {
            'annual_inc': float(request.form['annual_inc']),
            'revol_bal': float(request.form['revol_bal']),
            'emp_length': float(request.form['emp_length']),
            'open_acc': float(request.form['open_acc']),
            'home_ownership': request.form['home_ownership'],
            'grade': request.form['grade'],
            'purpose': request.form['purpose'],
            'addr_state': request.form['addr_state'],
            'initial_list_status': request.form['initial_list_status'],
            'application_type': request.form['application_type'],
            'verification_status_joint': request.form['verification_status_joint']
        }
        # Campos de fecha opcionales
        fechas = ['earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
        for campo in fechas:
            valor = request.form.get(campo, '').strip()
            if valor:
                datos_usuario[campo] = valor
        prob = crm.predict(datos_usuario)[0]
        score = int(crm.prob_to_score(prob))
        resultado = {
            'probabilidad': f"{prob*100:.2f}",
            'score': score
        }
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)