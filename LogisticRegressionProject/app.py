from flask import Flask, render_template, request
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    salary = int(request.form["salary"])

    data = scaler.transform([[age, salary]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Customer WILL purchase the product."
    else:
        result = "Customer will NOT purchase the product."

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)