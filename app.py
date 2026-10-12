from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved machine-learning model
model = joblib.load("random_forest_hardness_model.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form[name])
                for name in model.feature_names_in_
            ]

            data = pd.DataFrame(
                [values],
                columns=model.feature_names_in_
            )

            prediction = round(float(model.predict(data)[0]), 2)

        except Exception:
            error = "Please enter valid numerical values for every field."

    return render_template(
        "index.html",
        features=getattr(model, "feature_names_in_", []),
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run()
