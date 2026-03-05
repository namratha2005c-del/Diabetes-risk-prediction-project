from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model (make sure diabetes_model.pkl is in same folder)
model, scaler = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        bloodpressure = float(request.form["bloodpressure"])
        skinthickness = float(request.form["skinthickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])
        age = float(request.form["age"])

        features = np.array([[pregnancies, glucose, bloodpressure, skinthickness,
                              insulin, bmi, dpf, age]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "High Risk of Diabetes ⚠️"
        else:
            prediction[0] == 0
            result = "Low Risk of Diabetes ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)