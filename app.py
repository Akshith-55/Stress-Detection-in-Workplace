import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your trained machine learning models
models = {
    'random_forest': pickle.load(open('random_forest_model.pkl', 'rb'))
}

# Route for the home page
@app.route('/')
def home():
    return render_template("index.html")  # Flask looks in the "templates" folder

# Route to handle predictions
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract input values from form
        input_values = [
            float(request.form["Snoring Rate"]),
            float(request.form["Body Temperature"]),
            float(request.form["Blood Oxygen"]),
            float(request.form["Respiration Rate"]),
            float(request.form["Sleeping Hours"]),
            float(request.form["Heart Rate"]),
            int(request.form["Headache"]),
            int(request.form["Working Hours"])
        ]

        # Get the model selected by the user
        selected_model_key = request.form.get("model")
        model = models.get(selected_model_key)

        if model is None:
            return render_template("index.html", prediction_text="⚠️ Invalid model selected.")

        # Make prediction
        input_array = np.array([input_values])
        prediction = model.predict(input_array)

        # Map prediction result to stress levels
        stress_labels = {
            0: "Positive Stress",
            1: "Low Stress",
            2: "Medium Stress",
            3: "High Stress",
            4: "Extreme Stress"
        }

        result_text = stress_labels.get(prediction[0], "Unknown Stress Level")

        return render_template(
            "index.html",
            prediction_text=f"🧠 Stress level using {selected_model_key.replace('_', ' ').title()}: {result_text}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"🚫 Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
