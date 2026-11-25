# Red Wine Quality Prediction

Classifies red wines into **Low**, **Medium** or **High** quality categories from their physicochemical properties (acidity, sulphates, alcohol, etc.), using the UCI Wine Quality (red) dataset.

## Approach

1. **Exploration:** shape, dtypes, missing values, duplicates, quality score distribution.
2. **Cleaning:** removed duplicate rows (1,599 → 1,359 wines).
3. **Target:** grouped the 0–10 quality score into three classes:
   - Low: score ≤ 4
   - Medium: score 5–6
   - High: score ≥ 7
4. **Split:** 80/20 stratified train/test split (`random_state=42`).
5. **Class imbalance:** Medium makes up about 82% of the data, so I compared class weights and SMOTE oversampling.
6. **Models compared:** Logistic Regression, Random Forest, Random Forest with custom class weights, Gradient Boosting, SMOTE + Random Forest.
7. **Final model:** SMOTE + Random Forest in an `imblearn` Pipeline, evaluated with 5-fold stratified cross-validation on the training set and once on the held-out test set.

SMOTE is applied inside the pipeline, so it only touches training folds and never leaks into validation or test data.

## Results

| Metric | Cross-validation (train) | Test set |
|---|---|---|
| Accuracy | 0.755 | 0.757 |
| Macro F1 | 0.522 | 0.56 |

Final test classification report:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.33 | 0.38 | 0.36 | 13 |
| Medium | 0.89 | 0.80 | 0.85 | 222 |
| High | 0.40 | 0.62 | 0.48 | 37 |

**Limitations:** Low and High wines are rare (13 and 37 test samples), so per-class scores are noisy and macro F1 is only around 0.5. Accuracy alone is misleading here, because always predicting "Medium" scores about 82%. The most important features were alcohol, sulphates and volatile acidity.

## Project structure

```
.
├── Red_Wine_Quality_Prediction.ipynb
├── winequality-red.csv      # not tracked, download separately
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `winequality-red.csv` from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (check its licence and citation terms) and place it in `data/`. Then run the notebook:

```bash
jupyter notebook Red_Wine_Quality_Prediction.ipynb
```

Running the notebook also saves the trained pipeline as `wine_quality_model.pkl`.

## Using the saved model

```python
import joblib
import pandas as pd

model = joblib.load("wine_quality_model.pkl")

wine = pd.DataFrame([{
    "fixed acidity": 7.4, "volatile acidity": 0.30, "citric acid": 0.40,
    "residual sugar": 2.0, "chlorides": 0.060, "free sulfur dioxide": 20.0,
    "total sulfur dioxide": 45.0, "density": 0.9955, "pH": 3.25,
    "sulphates": 0.70, "alcohol": 11.5,
}])

print(model.predict(wine)[0])
print(dict(zip(model.classes_, model.predict_proba(wine)[0].round(3))))
```

**Notes:**
- Use the library versions in `requirements.txt`. A pickle saved with one scikit-learn version may fail or behave differently under another.
- Only load pickle files you trust, because loading one can execute arbitrary code.
