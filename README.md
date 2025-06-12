# Descripción de archivos del proyecto

## Archivos principales

- **web_app.py**: Implementa una aplicación web usando Flask. Permite a los usuarios ingresar datos financieros y personales a través de un formulario web, utiliza el modelo de riesgo de crédito para predecir la probabilidad de impago y calcular un score crediticio. Muestra los resultados en la página principal y almacena los resultados en una base de datos PostgreSQL.

- **app.py**: Implementa una interfaz de usuario usando Streamlit para calcular el score crediticio a partir de datos ingresados manualmente. Es útil para pruebas rápidas y visualización interactiva sin necesidad de un servidor web.

- **credit_risk_model.py**: Contiene la clase `CreditRiskModel`, que encapsula el flujo de preprocesamiento, entrenamiento, carga y predicción del modelo de riesgo de crédito. Utiliza TensorFlow/Keras para la red neuronal y joblib para el preprocesamiento. Incluye métodos para:
  - Preprocesar datos de entrada.
  - Entrenar el modelo a partir de un archivo CSV.
  - Guardar y cargar el modelo y las columnas usadas.
  - Realizar predicciones y convertir probabilidades en scores crediticios.

- **CreditRisk_Analisis_Modelo.ipynb**: Notebook de Jupyter que documenta y muestra paso a paso el análisis exploratorio, preprocesamiento, entrenamiento y evaluación del modelo de riesgo de crédito. Incluye visualizaciones, pruebas de modelos y explicación de cada etapa del pipeline de ciencia de datos.

- **modelo.ipynb**: Nuevo notebook de Jupyter que contiene experimentos adicionales, pruebas de selección de variables, análisis de importancia de características y flujos alternativos de entrenamiento y preprocesamiento del modelo.

## Scripts y utilidades

- **listar_columnas_modelo.py**: Script sencillo que carga el archivo preprocessing.pkl (donde se guardan las columnas usadas por el modelo) y las imprime por pantalla. Útil para revisar rápidamente qué variables espera el modelo.

- **test_db_connection.py**: (Vacío actualmente) Reservado para pruebas de conexión a base de datos.

## Archivos de datos y modelos

- **columnas_modelo.txt**: Lista de todas las columnas (features) utilizadas por el modelo de riesgo de crédito después del preprocesamiento.
- **credit_risk_model.h5**: Modelo de red neuronal entrenado (formato Keras/TensorFlow).
- **neural_network_model.h5**: Variante adicional de modelo de red neuronal.
- **reglog_model.pkl**: Modelo de regresión logística entrenado (pickle).
- **calibrator.pkl**: Archivo de calibración de probabilidades para modelos.
- **preprocessing.pkl**: Almacena la lista de columnas/features usadas por el modelo.
- **scaler.joblib**: Escalador de variables numéricas usado en el preprocesamiento.

## Visualizaciones y resultados

- **curva_entrenamiento.png**: Gráfica de la curva de pérdida (loss) durante el entrenamiento y validación de la red neuronal.
- **matriz_confusion_rna.png**: Matriz de confusión de la red neuronal, mostrando el desempeño del modelo en clasificación.

## Archivos de configuración y soporte

- **requirements.txt**: Lista de dependencias necesarias para ejecutar el proyecto (Flask, TensorFlow, pandas, numpy, joblib, scikit-learn, gspread, google-auth, psycopg2-binary).
- **Procfile**: Archivo para despliegue en plataformas como Heroku, indica el comando para iniciar la aplicación web.
- **.gitignore** y **.gitattributes**: Archivos de configuración de Git.

## Carpetas

- **templates/**: Plantillas HTML para la aplicación Flask (por ejemplo, index.html y style.css).
- **static/**: Archivos estáticos como hojas de estilo CSS.
- **loan/**: Carpeta donde se encuentra el archivo de datos loan.csv usado para entrenar el modelo (no se sube a GitHub por su tamaño).
- **__pycache__/**: Archivos compilados de Python (no se suben a GitHub).

## Otros archivos

- **LCDataDictionary.xlsx**: Diccionario de datos del dataset original.
- **prueba.txt**: Archivo de prueba sin relevancia para el funcionamiento del proyecto.
- **dbeaver-ce-25.1.0-x86_64-setup.exe**: Instalador de DBeaver, herramienta de administración de bases de datos (no necesario para el funcionamiento del proyecto).

---

# Instrucciones de uso y despliegue

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Para ejecutar la aplicación web (Flask):
   ```bash
   python web_app.py
   ```
3. Para ejecutar la interfaz Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Para entrenar el modelo desde cero, ejecuta `credit_risk_model.py` como script principal.

---

# Notas adicionales
- Las imágenes `curva_entrenamiento.png` y `matriz_confusion_rna.png` muestran el desempeño del modelo y pueden ser útiles para reportes o presentaciones.
- El archivo `modelo.ipynb` contiene experimentos y análisis adicionales que complementan el desarrollo del modelo principal.
- Los archivos binarios `.h5`, `.pkl`, `.joblib` son generados automáticamente durante el entrenamiento y pueden ser reemplazados si se reentrena el modelo.
