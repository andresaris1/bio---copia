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
**out_prncp**: Principal pendiente por pagar (monto del préstamo que aún no ha sido pagado). Ejemplo: 2000.
""")
out_prncp = st.number_input("Principal pendiente por pagar (out_prncp)", min_value=0.0, value=0.0)

st.markdown("""
**out_prncp_inv**: Principal pendiente a inversores (monto que aún se debe a los inversores). Ejemplo: 1500.
""")
out_prncp_inv = st.number_input("Principal pendiente a inversores (out_prncp_inv)", min_value=0.0, value=0.0)

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
        'out_prncp': out_prncp,
        'out_prncp_inv': out_prncp_inv,
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
