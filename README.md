# Documentación del Laboratorio 2 Despliegue de AutoML en Docker Automated Machine Learning (AutoML)


## Resolución del Proyecto

### **Estructura del Código**
El archivo principal (`Auto_ML.py`) incluye las siguientes funcionalidades clave:

1. **Cargar y Preprocesar Datos:**
    - Excluye columnas irrelevantes, como identificadores únicos.
    - Identifica columnas numéricas y categóricas, aplicando las transformaciones adecuadas mediante `ColumnTransformer`.

2. **Definir Modelos:**
    - Los modelos disponibles incluyen Random Forest, Gradient Boosting, SVM, KNN y Naive Bayes.
    - Permite optimización de hiperparámetros mediante `RandomizedSearchCV`.

3. **Pipeline de Entrenamiento:**
    - Combina preprocesamiento y modelo en un pipeline para facilitar la predicción.

4. **Evaluación:**
    - Genera un reporte de clasificación y una matriz de confusión para evaluar el rendimiento del modelo.

5. **Soporte para Despliegue Dual:**
    - API para predicciones en tiempo real utilizando `FastAPI`.
    - Batch Prediction para procesar archivos almacenados en una carpeta.


## Instrucciones de Ejecución

### **Requisitos Previos**
1. Instalar Docker.
2. Tener los archivos necesarios:
    - `Auto_ML.py`: Script principal.
    - `requirements.txt`: Dependencias necesarias.
    - `Dockerfile`: Archivo para construir la imagen Docker.
    - `.env`: Archivo de configuración de variables de entorno.
    - Dataset en formato Parquet.

---

### **Configuración del Archivo `.env`**

Ejemplo de configuración para **API**:
```env
DATASET=/data/Student_performance_data.parquet
TARGET=GradeClass
MODEL=RandomForest
TRIALS=10
DEPLOYMENT_TYPE=API
PORT=8000
```

Ejemplo de configuración para **Batch Prediction**:
```env
DATASET=/data/Student_performance_data.parquet
TARGET=GradeClass
MODEL=GradientBoosting
TRIALS=15
DEPLOYMENT_TYPE=Batch
INPUT_FOLDER=/data/input
OUTPUT_FOLDER=/data/output
```

---
### **Construcción del Dockerfile**

Ejemplo de configuración para **API**:
```
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
COPY data/Student_performance_data.parquet /data/Student_performance_data.parquet
CMD ["uvicorn", "Auto_ML:app","--host","0.0.0.0","--port", "8000"]

```

Ejemplo de configuración para **Batch Prediction**:
```
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
COPY data/Student_performance_data.parquet /data/Student_performance_data.parquet
CMD ["python", "Auto_ML.py"]

```

---

### **Construcción del Contenedor**

Ejecuta los siguientes comandos para construir y ejecutar el contenedor:


1. Construir la imagen Docker:
   ```bash
   docker build -t automl .
   ```

2. Ejecutar en modo API:
   ```bash
   docker run --env-file .env -p 8000:8000 automl
   ```

3. Ejecutar en modo Batch Prediction:
   ```bash
   docker run --env-file .env -v $(pwd)/data/input:/data/input -v $(pwd)/data/output:/data/output automl
   ```

---

### **Uso de la API**

Realiza una solicitud POST al endpoint `/predict` con un cuerpo JSON. Ejemplo:

`Usando Postman`
1. Abre Postman.
2. Selecciona el método POST.
3. En la URL, ingresa http://0.0.0.0:8000/predict.
4. En la pestaña Body, selecciona raw y establece el tipo como JSON. Debes llenarlo con los datos que deseas introducir al modelo para su predicción

```json
{
  "data": [
    {
      "Age": 17,
      "Gender": 1,
      "Ethnicity": 0,
      "ParentalEducation": 2,
      "StudyTimeWeekly": 19.83,
      "Absences": 7,
      "Tutoring": 1,
      "ParentalSupport": 2,
      "Extracurricular": 0,
      "Sports": 0,
      "Music": 1,
      "Volunteering": 0,
      "GPA": 2.93,
      "GradeClass": 2
    },
    {
      "Age": 18,
      "Gender": 0,
      "Ethnicity": 0,
      "ParentalEducation": 1,
      "StudyTimeWeekly": 15.41,
      "Absences": 0,
      "Tutoring": 0,
      "ParentalSupport": 1,
      "Extracurricular": 0,
      "Sports": 0,
      "Music": 0,
      "Volunteering": 0,
      "GPA": 3.04,
      "GradeClass": 1
    }
  ]
}

```
5. Haz clic en Send.

Ejemplo con `cURL`:
```bash
curl -X POST "http://0.0.0.0:8000/predict" -H "Content-Type: application/json" -d '{
    "data": [
        {
            "Age": 17,
            "Gender": 1,
            "Ethnicity": 0,
            "ParentalEducation": 2,
            "StudyTimeWeekly": 19.83,
            "Absences": 7,
            "Tutoring": 1,
            "ParentalSupport": 2,
            "Extracurricular": 0,
            "Sports": 0,
            "Music": 1,
            "Volunteering": 0,
            "GPA": 2.93,
            "GradeClass": 2
        },
        {
            "Age": 18,
            "Gender": 0,
            "Ethnicity": 0,
            "ParentalEducation": 1,
            "StudyTimeWeekly": 15.41,
            "Absences": 0,
            "Tutoring": 0,
            "ParentalSupport": 1,
            "Extracurricular": 0,
            "Sports": 0,
            "Music": 0,
            "Volunteering": 0,
            "GPA": 3.04,
            "GradeClass": 1
        }
    ]
}
```



---

### **Modo Batch Prediction**
1. Coloca los archivos Parquet en la carpeta de entrada (`/data/input`).
2. Los resultados se generarán en la carpeta de salida (`/data/output`) con el sufijo `_predictions.json`.

---

## Estructura de Archivos

```
.
├── Auto_ML.py          # Script principal
├── Dockerfile          # Archivo Docker
├── requirements.txt    # Dependencias del proyecto
├── .env                # Configuración de entorno
├── data/               # Carpeta de datos
│   ├── input/          # Carpeta para archivos de entrada (modo Batch)
│   └── output/         # Carpeta para resultados de predicciones (modo Batch)
```

