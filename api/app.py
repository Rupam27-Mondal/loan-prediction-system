from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model3.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        gender = request.form["Gender"]
        married = request.form["Married"]
        dependents = request.form["Dependents"]
        education = request.form["Education"]
        self_employed = request.form["Self_Employed"]
        credit_history = float(request.form["Credit_History"])
        property_area = request.form["Property_Area"]

        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_term = float(request.form["Loan_Amount_Term"])

        # 🔥 Encoding (same as training)
        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Yes" else 0
        education = 1 if education == "Graduate" else 0
        self_employed = 1 if self_employed == "Yes" else 0

        # Dependents encoding
        if dependents == "0":
            dependents = 0
        elif dependents == "1":
            dependents = 1
        elif dependents == "2":
            dependents = 2
        else:
            dependents = 3  # 3+

        # Property area encoding
        if property_area == "Urban":
            property_area = 2
        elif property_area == "Semiurban":
            property_area = 1
        else:
            property_area = 0

        # 🔥 Log transformations (VERY IMPORTANT)
        applicant_income_log = np.log(applicant_income + 1)
        loan_amount_log = np.log(loan_amount + 1)
        loan_term_log = np.log(loan_term + 1)
        total_income_log = np.log(applicant_income + coapplicant_income + 1)

        # Final feature array (11 features)
        features = np.array([[gender, married, dependents, education,
                              self_employed, credit_history, property_area,
                              applicant_income_log, loan_amount_log,
                              loan_term_log, total_income_log]])

        # Prediction
        prediction = model.predict(features)

        result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)