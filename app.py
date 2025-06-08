import streamlit as st
import pandas as pd
from credit_risk_model import CreditRiskModel

# Inicializar y cargar el modelo
crm = CreditRiskModel()
crm.load()

st.title("Scorecard de Riesgo Crediticio")

st.write("""
Ingrese sus características para conocer su score de riesgo crediticio.
""")

# Ejemplo de campos (ajusta según tus variables)
ingreso = st.number_input("Ingreso anual", min_value=0)
deuda = st.number_input("Deuda actual", min_value=0)
propietario = st.selectbox("¿Es propietario de vivienda?", ["RENT", "OWN", "MORTGAGE", "OTHER"])
grado = st.selectbox("Grado de crédito", ["A", "B", "C", "D", "E", "F", "G"])
# ... agrega más campos según tu modelo

if st.button("Calcular score"):
    # Construir diccionario con los datos del usuario
    datos_usuario = {
        'annual_inc': ingreso,
        'revol_bal': deuda,
        'home_ownership': propietario,
        'grade': grado,
        # ... agrega más campos aquí
    }
    # Predicción usando la clase
    prob = crm.predict(datos_usuario)[0]
    score = crm.prob_to_score(prob)

    st.success(f"Probabilidad de incumplimiento: {prob:.2%}")
    st.info(f"Su score crediticio es: {score:.0f} (rango 300-850)")

    # Scorecard visual (barra)
    st.progress((score-300)/550)

    # (Opcional) Comparar con la población
    # Puedes cargar un histograma de scores de la población y mostrarlo aquí
