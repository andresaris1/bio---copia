import streamlit as st
from credit_risk_model import CreditRiskModel

# Inicializar y cargar el modelo
crm = CreditRiskModel()
crm.load()

st.title("Scorecard de Riesgo Crediticio")

st.write("""
Ingrese los datos de su historial crediticio para conocer su score de riesgo.
""")

# Explicación de cada variable y campos de entrada
st.markdown("""
**last_pymnt_amnt**: Monto del último pago realizado por el cliente. Ejemplo: 500.
""")
last_pymnt_amnt = st.number_input("Monto del último pago (last_pymnt_amnt)", min_value=0.0, value=0.0)

st.markdown("""
**total_rec_prncp**: Total del principal recuperado (cuánto del préstamo original ya ha sido pagado). Ejemplo: 10000.
""")
total_rec_prncp = st.number_input("Total de principal recuperado (total_rec_prncp)", min_value=0.0, value=0.0)

st.markdown("""
**funded_amnt**: Monto total financiado al cliente (el préstamo otorgado). Ejemplo: 12000.
""")
funded_amnt = st.number_input("Monto financiado (funded_amnt)", min_value=0.0, value=0.0)

st.markdown("""
**funded_amnt_inv**: Monto financiado por los inversores (puede coincidir con el monto financiado si todo fue cubierto por inversores). Ejemplo: 12000.
""")
funded_amnt_inv = st.number_input("Monto financiado por inversores (funded_amnt_inv)", min_value=0.0, value=0.0)

st.markdown("""
**total_pymnt_inv**: Pagos totales realizados a los inversores. Ejemplo: 11000.
""")
total_pymnt_inv = st.number_input("Pagos totales a inversores (total_pymnt_inv)", min_value=0.0, value=0.0)

st.markdown("""
**total_pymnt**: Pagos totales realizados por el cliente (incluye intereses y principal). Ejemplo: 11000.
""")
total_pymnt = st.number_input("Pagos totales (total_pymnt)", min_value=0.0, value=0.0)

st.markdown("""
**recoveries**: Monto recuperado después de una pérdida o incumplimiento (si no aplica, poner 0). Ejemplo: 0.
""")
recoveries = st.number_input("Monto recuperado después de una pérdida (recoveries)", min_value=0.0, value=0.0)

if st.button("Calcular score"):
    datos_usuario = {
        'last_pymnt_amnt': last_pymnt_amnt,
        'total_rec_prncp': total_rec_prncp,
        'funded_amnt': funded_amnt,
        'funded_amnt_inv': funded_amnt_inv,
        'total_pymnt_inv': total_pymnt_inv,
        'total_pymnt': total_pymnt,
        'recoveries': recoveries
    }
    prob = crm.predict(datos_usuario)[0]
    score = crm.prob_to_score(prob)

    st.success(f"Probabilidad de incumplimiento: {prob:.2%}")
    st.info(f"Su score crediticio es: {score:.0f} (rango 300-850)")
    st.progress((score-300)/550)

    # (Opcional) Comparar con la población
    # Puedes cargar un histograma de scores de la población y mostrarlo aquí
