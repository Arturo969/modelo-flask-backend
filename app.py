# app.py
from flask import Flask, request, jsonify
import joblib          # ✅ Usamos joblib para cargar modelos ML
from flask_cors import CORS 
import numpy as np
import os              # Usamos 'os' para manejar rutas de archivos

# 1. Configuración de la aplicación
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir peticiones del frontend

# Variable global para el modelo y nombre del archivo
modelo = None
MODEL_FILE = 'knn_regression_model.pkl' # <-- ¡VERIFICA Y CAMBIA ESTO por el nombre exacto de tu archivo!

# 2. Cargar el modelo al iniciar (CRUCIAL: solo se ejecuta una vez)
try:
    # Verificamos si el archivo existe en la ruta actual
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"El archivo '{MODEL_FILE}' no se encontró en: {os.getcwd()}")
    
    # Cargamos el modelo usando joblib
    modelo = joblib.load(MODEL_FILE) 
    
    print(f"✅ Modelo '{MODEL_FILE}' cargado exitosamente usando joblib.")
except FileNotFoundError as fnf_e:
    print(f"❌ Error fatal al cargar el modelo: {fnf_e}")
    print("Por favor, asegúrate de que el archivo del modelo esté en la misma carpeta que app.py.")
    modelo = None # Dejamos el modelo como None para evitar errores si la carga falla.
except Exception as e:
    print(f"❌ Error al cargar el modelo (joblib): {e}")
    modelo = None

# 3. Ruta API para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Si el modelo no se cargó, devolvemos un error 500
    if modelo is None:
        return jsonify({'error': 'Modelo no disponible. Revisar logs del servidor.'}), 500

    try:
        # Obtener los datos (Latitud y Longitud) de la solicitud JSON
        data = request.get_json(force=True)
        latitud = data.get('latitude')
        longitud = data.get('longitude')

        if latitud is None or longitud is None:
            return jsonify({'error': 'Datos de entrada incompletos (requiere latitude y longitude)'}), 400

        # **ADAPTACIÓN CLAVE:**
        # Prepara los datos en el formato de array de NumPy, que es el esperado por scikit-learn/joblib
        features = np.array([[latitud, longitud]]) 
        
        # 4. Hacer la predicción
        prediction = modelo.predict(features)
        
        # Convertir el resultado a un tipo Python estándar para JSON (ej: float, int)
        # Usamos .item() para asegurarnos de que no sea un tipo numpy.float
        prediction_result = prediction[0].item() 

        # 5. Devolver la predicción
        return jsonify({'prediction': prediction_result})

    except Exception as e:
        # Captura errores que ocurren durante la predicción
        print(f"Error durante la predicción: {e}")
        return jsonify({'error': f'Error interno durante la predicción. Detalles: {e}'}), 500

# 6. Ejecutar el servidor
if __name__ == '__main__':
    # Usamos port=5000 por defecto
    app.run(debug=True)