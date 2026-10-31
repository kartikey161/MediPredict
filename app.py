from flask import Flask, request, jsonify, render_template
import json
import joblib
import pandas as pd
import numpy as np
import logging
import sqlite3

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the trained model bundle
bundle = joblib.load("p025_model_bundle.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
selected_features = bundle["selected_features"]
threshold = bundle["threshold"]

# Initialize SQLite database

def init_db():
    conn = sqlite3.connect("medipredict.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            patient_data TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            threshold REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/patient-form", methods=["GET"])
def patient_form():
    return render_template("patient_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list) or not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Convert JSON into DataFrame
        df = pd.DataFrame(data)

        # Check required features
        missing = [
            col for col in selected_features
            if col not in df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        # Maintain the exact training feature order
        df = df[selected_features].copy()

        # Convert values into numeric format
        df = df.apply(pd.to_numeric, errors="coerce")

        if df.isnull().any().any():
            return jsonify({
                "error": "Missing or invalid feature values",
                "invalid_features": df.columns[
                    df.isnull().any()
                ].tolist()
            }), 400

        if not np.isfinite(df.to_numpy(dtype=float)).all():
            return jsonify({
                "error": "Input contains infinite values"
            }), 400

        # Scale patient data
        scaled_data = scaler.transform(df)

        # Generate prediction probabilities
        probabilities = model.predict_proba(scaled_data)[:, 1]

        # Apply saved threshold
        predictions = (probabilities >= threshold).astype(int)

        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "threshold": threshold
        })

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 400

    
@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        form = request.form

        # 1. Read patient information

        def get_number(name, integer=False):
            value = form.get(name, "").strip()

            if not value:
                raise ValueError(f"{name} is required")

            number = float(value)

            if not np.isfinite(number):
                raise ValueError(f"{name} must be finite")

            if integer:
                if not number.is_integer():
                    raise ValueError(f"{name} must be a whole number")
                return int(number)

            return number

        age = get_number("age", True)
        gender = get_number("gender", True)

        time_in_hospital = get_number("time_in_hospital")
        num_lab_procedures = get_number("num_lab_procedures")
        num_procedures = get_number("num_procedures")
        num_medications = get_number("num_medications")

        number_outpatient = get_number("number_outpatient", True)
        number_emergency = get_number("number_emergency", True)
        number_inpatient = get_number("number_inpatient", True)
        number_diagnoses = get_number("number_diagnoses")

        encounter_count = get_number("encounter_count", True)
        change = get_number("change", True)

        # 2. Validate numerical inputs

        if not 1 <= age <= 10:
            raise ValueError("Invalid age group")

        if gender not in (0, 1):
            raise ValueError("Invalid gender")

        if any(value < 0 for value in [
            time_in_hospital, num_lab_procedures,
            num_procedures, num_medications,
            number_outpatient, number_emergency,
            number_inpatient
        ]):
            raise ValueError("Values cannot be negative")

        if number_diagnoses <= 0:
            raise ValueError("Number of diagnoses must be positive")

        if encounter_count < 1 or not 0 <= change <= encounter_count:
            raise ValueError("Invalid encounter or change count")

        # 3. Read medication information

        medication_names = [
            "metformin",
            "repaglinide",
            "glimepiride",
            "glipizide",
            "glyburide",
            "pioglitazone",
            "rosiglitazone",
            "insulin"
        ]

        medications = {}

        for med in medication_names:
            medications[med] = get_number(med, True)

            if medications[med] not in (0, 1):
                raise ValueError(f"Invalid value for {med}")

        # 4. Read categorical information

        specialty = form.get("medical_specialty_group")
        a1c = form.get("A1Cresult")

        diag_1 = form.get("diag_1_group")
        diag_2 = form.get("diag_2_group")
        diag_3 = form.get("diag_3_group")

        valid_specialties = [
            "Cardiology",
            "Family/GeneralPractice",
            "InternalMedicine",
            "Other"
        ]

        valid_a1c = ["Not Tested", "Norm", ">7", ">8"]

        valid_diagnoses = [
            "Diabetes", "Circulatory",
            "Respiratory", "Missing", "Other"
        ]

        if specialty not in valid_specialties:
            raise ValueError("Invalid medical specialty")

        if a1c not in valid_a1c:
            raise ValueError("Invalid A1C result")

        for diagnosis in [diag_1, diag_2, diag_3]:
            if diagnosis not in valid_diagnoses:
                raise ValueError("Invalid diagnosis")

        # 5. Automatic feature engineering

        service_utilization = (
            number_outpatient +
            number_emergency +
            number_inpatient
        )

        numchange = sum(medications.values())

        meds_per_diag = (
            num_medications / max(number_diagnoses, 1)
        )

        hospital_per_age = (
            time_in_hospital / max(age, 1)
        )

        # 6. Create all model features

        features = {
            "age": age,

            "time_in_hospital_log": np.log1p(time_in_hospital),
            "num_lab_procedures_log": np.log1p(num_lab_procedures),
            "num_procedures": num_procedures,
            "num_medications_log": np.log1p(num_medications),

            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient_log": np.log1p(number_inpatient),

            "number_diagnoses": number_diagnoses,

            "service_utilization_log": np.log1p(service_utilization),

            "numchange": numchange,
            "encounter_count": encounter_count,

            "meds_per_diag": meds_per_diag,
            "hospital_per_age": hospital_per_age,

            "num_medications|time_in_hospital":
                num_medications * time_in_hospital,

            "num_medications|number_diagnoses":
                num_medications * number_diagnoses,

            "medical_specialty_group_InternalMedicine":
                int(specialty == "InternalMedicine"),

            "medical_specialty_group_Other":
                int(specialty == "Other"),

            "A1Cresult_1":
                int(a1c in [">7", ">8"]),

            "diag_1_group_Other":
                int(diag_1 == "Other"),

            "diag_2_group_Diabetes":
                int(diag_2 == "Diabetes"),

            "diag_2_group_Other":
                int(diag_2 == "Other"),

            "diag_3_group_Diabetes":
                int(diag_3 == "Diabetes"),

            "diag_3_group_Other":
                int(diag_3 == "Other"),

            **medications,

            "gender": gender,
            "change": change
        }

        # 7. Arrange exactly 30 features in training order

        patient_df = pd.DataFrame(
            [[features[name] for name in selected_features]],
            columns=selected_features
        )

        # 8. Apply saved scaler

        scaled_data = scaler.transform(patient_df)

        # 9. Generate prediction

        probability = float(
            model.predict_proba(scaled_data)[0, 1]
        )

        prediction = int(probability >= threshold)

        # 10. Save prediction in SQLite

        with sqlite3.connect("medipredict.db") as conn:
            conn.execute("""
                INSERT INTO prediction_history
                (patient_data, prediction, probability, threshold)
                VALUES (?, ?, ?, ?)
            """, (
                json.dumps(form.to_dict()),
                prediction,
                probability,
                threshold
            ))

        # 11. Display result

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(probability * 100, 2),
            threshold=round(threshold * 100, 2)
        )

    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    except Exception:
        logging.exception("Patient form prediction failed")

        return jsonify({
            "error": "Unable to generate prediction. Check application logs."
        }), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        with open("dashboard_data.json", "r", encoding="utf-8") as file:
            stats = json.load(file)

        total_patients = sum(stats["readmission_values"])

        return render_template(
            "dashboard.html",
            stats=stats,
            total_patients=total_patients
        )

    except FileNotFoundError:
        return "Dashboard data file not found", 404


@app.route("/history", methods=["GET"])
def history():

    search = request.args.get("q", "").strip()
    status = request.args.get("status", "all")

    query = """
        SELECT id, created_at, prediction, probability, threshold
        FROM prediction_history
        WHERE 1=1
    """

    params = []

    # Filter by prediction status
    if status == "readmitted":
        query += " AND prediction = 1"

    elif status == "not_readmitted":
        query += " AND prediction = 0"

    # Search by ID or date
    if search:
        if search.isdigit():
            query += " AND CAST(id AS TEXT) LIKE ?"
            params.append(f"%{search}%")
        else:
            query += " AND created_at LIKE ?"
            params.append(f"%{search}%")

    query += " ORDER BY id DESC"

    with sqlite3.connect("medipredict.db") as conn:
        conn.row_factory = sqlite3.Row
        records = conn.execute(query, params).fetchall()

    return render_template(
        "history.html",
        records=records,
        search=search,
        status=status
    )

@app.route("/history/<int:record_id>", methods=["GET"])
def history_detail(record_id):

    with sqlite3.connect("medipredict.db") as conn:
        conn.row_factory = sqlite3.Row

        record = conn.execute("""
            SELECT *
            FROM prediction_history
            WHERE id = ?
        """, (record_id,)).fetchone()

    if record is None:
        return "Prediction record not found", 404

    patient_data = json.loads(record["patient_data"])

    return render_template(
        "history_detail.html",
        record=record,
        patient_data=patient_data
    )


if __name__ == "__main__":
    init_db()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )