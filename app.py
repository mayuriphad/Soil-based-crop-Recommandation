from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ✅ Load Trained Crop Model
model = joblib.load("crop_recommendation_model.pkl")

# ✅ Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # 📥 Get form data
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # 🔍 Predict
            features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            crop = model.predict(features)[0]
            prediction = f"🌾 Recommended Crop: {crop.capitalize()}"

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
