from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3
import json
import warnings

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Cambio de directorio al lugar donde se encuentra el script
os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

#1
@app.route('/v2/predict', methods=['GET'])
def predict():
    # Usar la ruta completa para cargar el modelo pickle
    model_path = 'C:\\Users\\Admin\\Desktop\\App_Model\\ejercicio\\data\\advertising_model'
    model = pickle.load(open(model_path, 'rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        # Ajustar nombres de características y capitalización
        feature_names = ['TV', 'radio', 'newspaper']
        input_data = pd.DataFrame([[int(tv), int(radio), int(newspaper)]], columns=feature_names)
        prediction = model.predict(input_data)
        return jsonify({"prediction": round(prediction[0], 2)})

#2
@app.route('/v2/ingest_data', methods=['POST'])
def ingest_data():
    # Usar la ruta completa para la base de datos SQLite
    db_path = 'C:\\Users\\Admin\\Desktop\\App_Model\\data\\advertising.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    sales = request.args.get('sales', None)

    query = '''INSERT INTO campañas VALUES (?, ?, ?, ?)'''
    query_2 = "SELECT * FROM campañas"

    cursor.execute(query, (tv, radio, newspaper, sales))
    result = cursor.execute(query_2).fetchall()

    connection.commit()
    connection.close()

    return jsonify(result)

#3
@app.route('/v2/retrain', methods=['PUT'])
def retrain():
    # Usar la ruta completa para la base de datos SQLite
    db_path = 'C:\\Users\\Admin\\Desktop\\App_Model\\data\\advertising.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = '''SELECT * FROM campañas'''
    result = cursor.execute(query).fetchall()

    columns = [columns[0] for columns in cursor.description]

    df = pd.DataFrame(data=result, columns=columns)
    df.dropna(inplace=True)

    X = df.drop(columns=('sales'))
    Y = df['sales']

    # Usar la ruta completa para cargar y guardar el modelo pickle
    model_path = 'C:\\Users\\Admin\\Desktop\\App_Model\\ejercicio\\data\\advertising_model'
    model = pickle.load(open(model_path, 'rb'))
    mae_1 = - (cross_val_score(model, X, Y, cv=5, scoring='neg_mean_absolute_error')).mean()

    model.fit(X, Y)
    mae_2 = - (cross_val_score(model, X, Y, cv=5, scoring='neg_mean_absolute_error')).mean()

    if mae_2 < mae_1:
        pickle.dump(model, open(model_path, 'wb'))
        return f'Your new model is better, mae = {mae_2}'
    else:
        return f'Your old model is better, mae = {mae_1}'

if __name__ == '__main__':
    # Información a incluir en el JSON
    info_json = {
        "repository": "https://github.com/D-AZ-SE/app_model.git",
        "url": "https://azpitarte.pythonanywhere.com/"
    }

    # Ruta completa del archivo JSON
    json_file_path = os.path.join(os.path.dirname(__file__), 'entregas', 'entregable_daniel.json')

    print(f"Ruta completa del archivo JSON: {json_file_path}")
    try:
        # Guardar el JSON en el archivo
        with open(json_file_path, "w") as json_file:
            json.dump(info_json, json_file, indent=2)

        print(f"JSON guardado en: {json_file_path}")

    except Exception as e:
        print(f"Error al guardar el JSON: {e}")

    app.run(debug=False)