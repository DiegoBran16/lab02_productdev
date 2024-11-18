import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import joblib
import time
import glob

#cargando las variables de entorno
load_dotenv()

load_dotenv()

deployment_type = os.getenv('DEPLOYMENT_TYPE')
data_path = os.getenv('DATASET')
target_column = os.getenv('TARGET')
model_name = os.getenv('MODEL')
trials = int(os.getenv('TRIALS'))

if deployment_type == "API":
    port = int(os.getenv('PORT', 8000))
elif deployment_type == "Batch":
    input_folder = os.getenv('INPUT_FOLDER', './input')
    output_folder = os.getenv('OUTPUT_FOLDER', './output')

# Cargando y prerocesando el dataset
data = pd.read_parquet(data_path)

# Excluir columnas con valores únicos (como IDs)
unique_threshold = len(data)  # Número total de filas
columns_to_exclude = [col for col in data.columns if data[col].nunique() == unique_threshold]

# Evitar excluir 'target_column'
columns_to_exclude = [col for col in columns_to_exclude if col != target_column]

# Excluir las columnas identificadas
X = data.drop(columns=columns_to_exclude + [target_column])
y = data[target_column]

#identificar columnas numericas y categoricas

nun_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object','category','string']).columns.tolist()

# preprocesar el pipeline
preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), nun_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ]
    )
# Definicion de modelos
models = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
    }

if model_name not in models:
    raise ValueError(f"El modelo {model_name} no esta disponible, elige uno de estos: {list(models.keys())}")

model = models[model_name]

# Optimización de Hyperparametros

param_distributions = {
        "RandomForest":{
            "n_estimators":[10,50,100,200],
            "max_depth":[None, 10,20,30],
            "min_samples_split":[2,5,10]
        },
        "GradientBoosting":{
            "n_estimators":[50,100,200],
            "learning_rate":[0.01,0.1,0.2],
            "max_depth":[2,5,10]
        },
        "SVM":{
            "C": [0.1,1,10],
            "gamma": [0.001,0.01,0.1]
        },
        "KNN":{
            "n_neighbors":[3,5,7],
            "weights":['uniform','distance']
        },
        "NaiveBayes":{}
}

search = RandomizedSearchCV(model, param_distributions.get(model_name,{}), n_iter=trials, cv=5, random_state=42)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', search)])
pipeline.fit(X_train, y_train)

#evaluar el modelo
if isinstance(pipeline.named_steps['model'], RandomizedSearchCV):
    best_model = pipeline.named_steps['model'].best_estimator_
else:
    best_model = pipeline.named_steps['model']

y_pred = pipeline.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Guardar el modelo
joblib.dump(pipeline, f'{model_name}.pkl')

#Implementación de la API
app = FastAPI()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame(request.data)
        predictions = pipeline.predict_proba(input_data)
        result = [dict(zip(pipeline.classes_, pred)) for pred in predictions]
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Batch Prediction
def batch_prediction(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    while True:
        files = glob.glob(f"{input_folder}/*.parquet")
        for file_path in files:
            try:
                # Cargar el archivo Parquet
                input_data = pd.read_parquet(file_path)
                # Realizar predicciones
                predictions = pipeline.predict_proba(input_data)
                result = [dict(zip(pipeline.classes_, pred)) for pred in predictions]

                # Crear archivo de salida
                output_file = file_path.replace(input_folder, output_folder).replace(".parquet", "_predictions.json")
                with open(output_file, 'w') as f:
                    json.dump({"predictions": result}, f, indent=4)

                print(f"Procesado: {file_path} -> {output_file}")
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")
        time.sleep(10)  # Esperar 10 segundos antes de buscar nuevos archivos

# Ejecutar la app
def main():
    if deployment_type == "API":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    elif deployment_type == "Batch":
        batch_prediction(input_folder, output_folder)
    else:
        raise ValueError("DEPLOYMENT_TYPE debe ser 'API' o 'Batch'")

if __name__ == "__main__":
    main()

