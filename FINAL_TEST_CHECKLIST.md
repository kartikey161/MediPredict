# MediPredict — Final Test Checklist

Use this checklist from the project root with the `.venv` environment activated.

## A. Startup and syntax

- [ ] Run `python -m py_compile app.py` — it should produce no output.
- [ ] Run `python app.py` — Flask should start on `http://127.0.0.1:5000`.
- [ ] Confirm the server terminal has no traceback.

## B. Pages

- [ ] `/` loads the homepage.
- [ ] `/patient-form` loads the patient form.
- [ ] `/dashboard` loads the four Plotly charts and summary cards.
- [ ] `/history` loads saved prediction records (or an empty-state message).

## C. End-to-end prediction and database

- [ ] Submit a valid synthetic/de-identified test record through the form.
- [ ] Confirm the result page shows predicted class, probability and threshold.
- [ ] Confirm a new row appears in `/history`.
- [ ] Confirm another submission creates another database row.
- [ ] Compare a known held-out notebook record with the web form. The tested record previously matched at 80.53% in both places; repeat this check after any feature-engineering changes.
- [ ] Try an invalid/missing form value and confirm the app returns a controlled error rather than a traceback page.

To count saved rows:

```powershell
python -c "import sqlite3; c=sqlite3.connect('medipredict.db'); print(c.execute('SELECT COUNT(*) FROM prediction_history').fetchone()[0]); c.close()"
```

## D. Before pushing to GitHub

- [ ] Confirm `p025_model_bundle.joblib` and `dashboard_data.json` are present in the project root.
- [ ] Do not commit `.venv/`, `medipredict.db`, `app.log`, `diabetic_data.csv`, `test.json`, `test_preprocessed.json`, or `predictions.csv`.
- [ ] Review notebook outputs for personal, private, or unintended data.
- [ ] Keep the UCI dataset citation in `README.md`.
- [ ] Verify the README commands work in a fresh environment if possible.

## Scope

No additional application features are required for the minimal academic version. Focus on fixing test failures, keeping the existing six core features stable, and documenting the project accurately.
