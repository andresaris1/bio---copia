import streamlit as st
import pandas as pd
from credit_risk_model import CreditRiskModel

# Inicializar y cargar el modelo
crm = CreditRiskModel()
crm.load()

st.title("Scorecard de Riesgo Crediticio")

st.write("""
Ingrese los datos de su historial crediticio para conocer su score de riesgo.
""")

# Campos de la encuesta (las 9 variables principales)
last_pymnt_amnt = st.number_input("Monto del último pago (last_pymnt_amnt)", min_value=0.0)
total_rec_prncp = st.number_input("Total de principal recuperado (total_rec_prncp)", min_value=0.0)
funded_amnt = st.number_input("Monto financiado (funded_amnt)", min_value=0.0)
funded_amnt_inv = st.number_input("Monto financiado por inversores (funded_amnt_inv)", min_value=0.0)
total_pymnt_inv = st.number_input("Pagos totales a inversores (total_pymnt_inv)", min_value=0.0)
total_pymnt = st.number_input("Pagos totales (total_pymnt)", min_value=0.0)
total_rec_int = st.number_input("Total de intereses recuperados (total_rec_int)", min_value=0.0)
out_prncp_inv = st.number_input("Principal pendiente a inversores (out_prncp_inv)", min_value=0.0)
out_prncp = st.number_input("Principal pendiente (out_prncp)", min_value=0.0)

if st.button("Calcular score"):
    datos_usuario = {
        'last_pymnt_amnt': last_pymnt_amnt,
        'total_rec_prncp': total_rec_prncp,
        'funded_amnt': funded_amnt,
        'funded_amnt_inv': funded_amnt_inv,
        'total_pymnt_inv': total_pymnt_inv,
        'total_pymnt': total_pymnt,
        'total_rec_int': total_rec_int,
        'out_prncp_inv': out_prncp_inv,
        'out_prncp': out_prncp,
    }
    prob = crm.predict(datos_usuario)[0]
    score = crm.prob_to_score(prob)

    st.success(f"Probabilidad de incumplimiento: {prob:.2%}")
    st.info(f"Su score crediticio es: {score:.0f} (rango 300-850)")
    st.progress((score-300)/550)

    # (Opcional) Comparar con la población
    # Puedes cargar un histograma de scores de la población y mostrarlo aquí
