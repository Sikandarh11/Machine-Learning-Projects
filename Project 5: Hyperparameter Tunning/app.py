import keras
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
app = Flask(__name__)
try:
    model =keras.models.load_model("D:\\heart_disease_model.h5")
    scaler = joblib.load("D:\\scaler.pkl")
except FileNotFoundError:
    print("Model file not found. Ensure the path is correct.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.form
        age = data.get("age")
        Sex= data.get("sex")
        Chest_pain_type= data.get("chest_pain_type" )
        BP = data.get("bp" )
        Cholesterol= data.get("cholesterol" )
        FBS_over_120= data.get("fbs")
        EKG_results= data.get("ekg_results" )
        Max_HR= data.get("max_hr")
        Exercise_angina= data.get("exercise_angina")
        ST_depression= data.get("st_depression" )
        Slope_of_ST= data.get("slope_of_st")
        Number_of_vessels_fluro= data.get("num_vessels_fluro")
        Thallium= data.get("thallium")
        input_data=[age,Sex, Chest_pain_type, BP, Cholesterol, FBS_over_120, EKG_results, Max_HR, Exercise_angina
                    ,ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium]
        if input_data is None:
            return jsonify({"error": "No input data provided"}), 400

        data = np.array(input_data).reshape(1, -1)

        if model.predict(scaler.transform(data)) >= 0.5:
            result="This patient has heart disease"
        else:
            result = "This patient does not have heart disease"
        return render_template("result.html", result =result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
