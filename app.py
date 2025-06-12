import streamlit as st
from credit_risk_model import CreditRiskModel

# Inicializar y cargar el modelo
crm = CreditRiskModel()
crm.load()

st.title("Scorecard de Riesgo Crediticio")

st.write("""
Ingrese los datos de su historial crediticio y comportamiento de pagos para conocer su score de riesgo.
""")

st.markdown("""
**pct_principal_paid**: Porcentaje del principal pagado respecto al monto original del préstamo. Ejemplo: 0.8 para 80% pagado.
""")
pct_principal_paid = st.number_input("Porcentaje del principal pagado (pct_principal_paid)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

st.markdown("""
**total_rec_prncp**: Total del principal recuperado (cuánto del préstamo original ya ha sido pagado). Ejemplo: 10000.
""")
total_rec_prncp = st.number_input("Total de principal recuperado (total_rec_prncp)", min_value=0.0, value=0.0)

st.markdown("""
**last_pymnt_amnt**: Monto del último pago realizado por el cliente. Ejemplo: 500.
""")
last_pymnt_amnt = st.number_input("Monto del último pago (last_pymnt_amnt)", min_value=0.0, value=0.0)

st.markdown("""
**recoveries**: Monto recuperado después de una pérdida o incumplimiento (si no aplica, poner 0). Ejemplo: 0.
""")
recoveries = st.number_input("Monto recuperado después de una pérdida (recoveries)", min_value=0.0, value=0.0)

st.markdown("""
**total_pymnt**: Pagos totales realizados por el cliente (incluye intereses y principal). Ejemplo: 11000.
""")
total_pymnt = st.number_input("Pagos totales (total_pymnt)", min_value=0.0, value=0.0)

st.markdown("""
**total_pymnt_inv**: Pagos totales realizados a los inversores. Ejemplo: 11000.
""")
total_pymnt_inv = st.number_input("Pagos totales a inversores (total_pymnt_inv)", min_value=0.0, value=0.0)

st.markdown("""
**out_prncp**: Principal pendiente por pagar (monto del préstamo que aún no ha sido pagado). Ejemplo: 2000.
""")
out_prncp = st.number_input("Principal pendiente por pagar (out_prncp)", min_value=0.0, value=0.0)

st.markdown("""
**out_prncp_inv**: Principal pendiente a inversores (monto que aún se debe a los inversores). Ejemplo: 1500.
""")
out_prncp_inv = st.number_input("Principal pendiente a inversores (out_prncp_inv)", min_value=0.0, value=0.0)

st.markdown("""
**int_rate**: Tasa de interés anual del préstamo (en porcentaje). Ejemplo: 12.5 para 12.5%.
""")
int_rate = st.number_input("Tasa de interés anual (int_rate)", min_value=0.0, max_value=100.0, value=0.0, step=0.01)

st.markdown("""
**pct_term_paid**: Porcentaje del plazo del préstamo que ya ha transcurrido. Ejemplo: 0.5 para 50% del plazo.
""")
pct_term_paid = st.number_input("Porcentaje del plazo pagado (pct_term_paid)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

if st.button("Calcular score"):
    datos_usuario = {
        'pct_principal_paid': pct_principal_paid,
        'total_rec_prncp': total_rec_prncp,
        'last_pymnt_amnt': last_pymnt_amnt,
        'recoveries': recoveries,
        'total_pymnt': total_pymnt,
        'total_pymnt_inv': total_pymnt_inv,
        'out_prncp': out_prncp,
        'out_prncp_inv': out_prncp_inv,
        'int_rate': int_rate,
        'pct_term_paid': pct_term_paid
    }
    prob = crm.predict(datos_usuario)[0]
    score = crm.prob_to_score(prob)

    st.success(f"Probabilidad de incumplimiento: {prob:.2%}")
    st.info(f"Su score crediticio es: {score:.0f} (rango 300-850)")
    st.progress((score-300)/550)

    # (Opcional) Comparar con la población
    # Puedes cargar un histograma de scores de la población y mostrarlo aquí
