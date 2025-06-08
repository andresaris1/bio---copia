Descripción de archivos del proyecto

web_app.py

Este archivo implementa una aplicación web usando Flask. Permite a los usuarios ingresar datos financieros y personales a través de un formulario web, y utiliza el modelo de riesgo de crédito para predecir la probabilidad de impago y calcular un score crediticio. Muestra los resultados en la página principal.

credit_risk_model.py

Contiene la clase CreditRiskModel, que encapsula todo el flujo de preprocesamiento, entrenamiento, carga y predicción del modelo de riesgo de crédito. Utiliza TensorFlow/Keras para la red neuronal y joblib para guardar/cargar el preprocesamiento. 
Incluye métodos para:

Preprocesar datos de entrada.

Entrenar el modelo a partir de un archivo CSV.

Guardar y cargar el modelo y las columnas usadas.

Realizar predicciones y convertir probabilidades en scores crediticios.

CreditRisk_Analisis_Modelo.ipynb

Notebook de Jupyter que documenta y muestra paso a paso el análisis exploratorio, preprocesamiento, entrenamiento y evaluación del modelo de riesgo de crédito. Incluye visualizaciones, pruebas de modelos y explicación de cada etapa del pipeline de ciencia de datos.

listar_columnas_modelo.py

Script sencillo que carga el archivo preprocessing.pkl (donde se guardan las columnas usadas por el modelo) y las imprime por pantalla. Útil para revisar rápidamente qué variables espera el modelo.

columnas_modelo.txt

Archivo de texto que contiene la lista de todas las columnas (features) utilizadas por el modelo de riesgo de crédito después del preprocesamiento. Sirve como referencia para saber qué variables deben estar presentes en los datos de entrada.

credit_risk_model.h5

Archivo binario que contiene el modelo de red neuronal entrenado (formato Keras/TensorFlow). No se sube a GitHub si está en .gitignore.

preprocessing.pkl

Archivo binario generado con joblib que almacena la lista de columnas/features usadas por el modelo, para asegurar que los datos de entrada tengan el mismo formato que los usados en el entrenamiento.

templates/

Carpeta que contiene las plantillas HTML para la aplicación Flask. Por ejemplo, index.html define la interfaz web para el formulario de entrada y la visualización de resultados.

loan/

Carpeta donde originalmente se encontraba el archivo de datos loan.csv usado para entrenar el modelo. Este archivo no se sube a GitHub por su gran tamaño.

LCDataDictionary.xlsx

Archivo de Excel que probablemente contiene el diccionario de datos, es decir, la descripción de cada variable presente en el dataset original.

__pycache__/

Carpeta generada automáticamente por Python para almacenar archivos compilados. Está en .gitignore y no se sube a GitHub.
