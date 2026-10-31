# MediPredict — 30-Day Hospital Readmission Prediction

MediPredict is an academic machine-learning web application that demonstrates how historical hospital encounter data can be used to estimate whether a patient record belongs to the dataset's **readmission-within-30-days** class. It uses a Flask backend, a saved XGBoost model bundle, an HTML/CSS interface, Plotly charts, and SQLite for prediction history.

> **Research prototype only.** This application is not a medical device, diagnostic system, or clinical decision-support tool. Its predictions are not clinically validated and must not be used to make patient-care decisions.

## Features

- Homepage describing the project.
- Patient information form.
- Automatic calculation of the selected model features and application of the saved `StandardScaler`.
- XGBoost prediction result with probability and the configured classification threshold.
- Historical dataset analytics dashboard with Plotly charts.
- SQLite history of submitted predictions.

## Technology stack

- Python 3.13 (the development environment used for this project)
- Flask
- pandas and NumPy
- scikit-learn and imbalanced-learn
- XGBoost
- Plotly
- SQLite
- HTML and CSS
- Jupyter Notebook for data preparation and model training

## Project structure

```text
prediction-on-hospital-readmission/
├── app.py
├── requirements.txt
├── README.md
├── p025_model_bundle.joblib
├── dashboard_data.json
├── diabetic_data.csv              # Download separately; do not commit by default
├── medipredict.db                 # Created automatically; do not commit
├── prediction-on-hospital-readmission.ipynb
├── templates/
│   ├── index.html
│   ├── patient_form.html
│   ├── result.html
│   ├── dashboard.html
│   └── history.html
└── static/
    └── css/
        └── style.css
```

The application needs `p025_model_bundle.joblib` and `dashboard_data.json` in the project root. The database is created by `init_db()` when the Flask application starts. The raw CSV is needed for notebook-based data preparation/training, not for serving the already-created model and dashboard files.

## Setup on Windows PowerShell

Open a terminal in the project directory, then run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If the project already has a working `.venv`, activate it instead of recreating it.

## Run the application

From the project root, with `.venv` activated:

```powershell
python -m py_compile app.py
python app.py
```

Open these local routes in a browser:

| Page / endpoint | URL | Purpose |
|---|---|---|
| Home | `http://127.0.0.1:5000/` | Project landing page |
| Patient form | `http://127.0.0.1:5000/patient-form` | Enter the aggregated patient-record inputs |
| Prediction | `POST /predict-form` | Process the form and show the result |
| JSON API | `POST /predict` | Accept already-prepared model features as JSON |
| Analytics | `http://127.0.0.1:5000/dashboard` | Historical dataset charts |
| History | `http://127.0.0.1:5000/history` | Previously saved predictions |

Stop the development server with `Ctrl+C`.

## Model and data notes

- The original UCI Diabetes 130-US Hospitals dataset contains 101,766 encounter rows and 50 columns in the downloaded CSV used during development.
- The notebook aggregates encounters by patient, producing 70,443 patient-level rows in the current run.
- The model uses 30 selected features. The train/test split used 56,354 training rows and 14,089 test rows.
- The saved bundle `p025_model_bundle.joblib` contains the fitted XGBoost classifier, fitted scaler, selected feature names, and decision threshold (0.5).
- The notebook maps `<30` to class `1`; `>30` and `NO` are both mapped to class `0`. Therefore, class `0` means **not labelled as readmitted within 30 days**, and does not necessarily mean the patient was never readmitted.
- The patient form accepts patient-level summary values. The feature calculations in `app.py` should be kept consistent with the notebook's feature engineering.

### Reported hold-out evaluation

The current notebook run reported the following classification results (rounded):

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 — not readmitted within 30 days | 0.96 | 0.87 | 0.91 | 12,322 |
| 1 — readmitted within 30 days | 0.45 | 0.74 | 0.56 | 1,767 |
| Overall accuracy | | | 0.85 | 14,089 |

Confusion matrix, with rows as actual class and columns as predicted class:

```text
[[10726, 1596],
 [  452, 1315]]
```

The class-1 precision is relatively low, so many positive predictions are false positives. Accuracy alone does not describe model usefulness. These figures are from a held-out split of this dataset, not from an external clinical validation study.

## Dataset source

The dataset is the UCI Machine Learning Repository's **Diabetes 130-US Hospitals for Years 1999–2008** dataset:

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Download and extract the dataset. Place `diabetic_data.csv` in the project root before running the notebook. Follow the dataset's citation and usage terms when redistributing it.

## Important repository hygiene

Do not commit the virtual environment, SQLite database, logs, caches, or test patient JSON files. The database may contain submitted form values. Do not put real identifiable patient information into this academic prototype or a public GitHub repository.

Before publishing, check that `p025_model_bundle.joblib` and `dashboard_data.json` are present, that no private data is included, and that your repository's licence and dataset citation are appropriate.
