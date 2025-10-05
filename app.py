from flask import Flask, request, jsonify
import joblib 
from flask_cors import CORS 
import numpy as np
import os

# 1. Configuración y Archivos de Modelos
app = Flask(__name__)
CORS(app) 

# Nombres de los archivos de modelos
MODEL_FILE_A = 'knn_population_model.pkl' 
MODEL_FILE_B = 'knn_regression_model.pkl' 

# Variables globales para los modelos
modelo_A = None
modelo_B = None

# Función utilitaria para cargar un modelo específico
def load_model(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo '{file_path}' no se encontró en: {os.getcwd()}")
        
        model = joblib.load(file_path)
        print(f"✅ Modelo '{file_path}' cargado exitosamente usando joblib.")
        return model
    except Exception as e:
        print(f"❌ Error al cargar el modelo {file_path}: {e}")
        return None

# 2. Cargar AMBOS modelos al iniciar la aplicación
modelo_A = load_model(MODEL_FILE_A)
modelo_B = load_model(MODEL_FILE_B)


# Función utilitaria para realizar la predicción
def make_prediction(model, data):
    """Procesa los datos y hace la predicción con el modelo dado."""
    if model is None:
        return {'error': 'Modelo no disponible en el servidor.'}, 500

    try:
        latitud = data.get('latitude')
        longitud = data.get('longitude')

        if latitud is None or longitud is None:
            return {'error': 'Datos de entrada incompletos (requiere latitude y longitude)'}, 400

        # Prepara los datos en el formato de array de NumPy
        features = np.array([[latitud, longitud]]) 
        
        # Hacer la predicción
        prediction = model.predict(features)
        
        # Convertir el resultado a un tipo Python estándar para JSON
        prediction_result = prediction[0].item() 

        # Devolver la predicción
        return {'prediction': prediction_result}, 200

    except Exception as e:
        # Captura errores que ocurren durante la predicción (ej. formato de datos incorrecto)
        print(f"Error durante la predicción: {e}")
        return {'error': f'Error interno durante la predicción. Detalles: {e}'}, 500


# 3. Ruta API para el Modelo A (/predict/A)
@app.route('/predict/A', methods=['POST'])
def predict_A():
    data = request.get_json(force=True)
    result, status = make_prediction(modelo_A, data)
    return jsonify(result), status

# 4. Ruta API para el Modelo B (/predict/B)
@app.route('/predict/B', methods=['POST'])
def predict_B():
    data = request.get_json(force=True)
    result, status = make_prediction(modelo_B, data)
    return jsonify(result), status


# 5. Ejecutar el servidor (Sólo para desarrollo local)
if __name__ == '__main__':
    app.run(debug=True)
