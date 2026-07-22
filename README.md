# House Price Prediction — Supervised Learning (Regression)

End-to-end regression pipeline predicting residential `SalePrice` on the **Ames Housing Dataset**,
built as a Practical Exam (Set A) submission for a PropTech-style house price estimator.

## Dataset

- Source: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- File used: `train.csv` (1,460 rows, 79 features, target = `SalePrice`)
- Only `train.csv` is used; the exam requires a custom train/test split (80/20, `random_state=42`).

## Project Structure

```
.
├── HousePrice_SupervisedLearning.ipynb   # Fully executed notebook (EDA -> preprocessing -> models -> evaluation)
├── house_price_model.pkl                 # Saved sklearn Pipeline (preprocessing + tuned XGBoost)
├── summary_report.md                     # ~400-500 word write-up of approach & findings
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## What the Notebook Does

1. **Theory notes** — regression vs classification, Linear/Ridge/Lasso, overfitting/underfitting,
   RMSE/MAE/R², k-Fold cross-validation.
2. **EDA** — target distribution & log1p transform, skewness, univariate/bivariate plots,
   correlation heatmap, outlier flagging.
3. **Preprocessing & Feature Engineering** — missing value strategy, outlier removal, engineered
   features (`TotalSF`, `HouseAge`, `RemodAge`, `HasGarage`, `HasPool`), ordinal encoding, one-hot
   encoding, target-mean encoding for `Neighborhood`, skew correction, scaling — all wrapped in a
   `ColumnTransformer` + `Pipeline`.
4. **Models** — Linear Regression, RidgeCV, LassoCV, Random Forest, XGBoost, plus 5-fold CV and
   `RandomizedSearchCV` tuning on XGBoost.
5. **Model comparison** — RMSE / MAE / R² / CV RMSE / training time table (all metrics in original
   USD scale via `expm1`), with the tuned XGBoost selected as the best model.
6. **Residual analysis** — residuals vs fitted, residual histogram, Q-Q plot, plain-English
   business interpretation of the top predictive features.
7. **Deployment** — final pipeline saved via `joblib`, reloaded, and tested on 5 sample rows.

## How to Run

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/house-price-regression-supervised-learning.git
cd house-price-regression-supervised-learning

# 2. Create environment & install dependencies
pip install -r requirements.txt

# 3. Place train.csv (from the Kaggle link above) in this folder

# 4. Launch Jupyter and run all cells
jupyter notebook HousePrice_SupervisedLearning.ipynb
```

To use the saved model directly:

```python
import joblib
import pandas as pd
import numpy as np

pipeline = joblib.load("house_price_model.pkl")
predicted_log_price = pipeline.predict(new_data_df)   # new_data_df: same columns as X_train
predicted_price = np.expm1(predicted_log_price)       # convert back to USD
```

## Results Summary

| Model               | RMSE (USD) | MAE (USD) | R²    |
|---------------------|-----------:|----------:|------:|
| XGBoost (Tuned)     | ~19,332    | ~14,104   | 0.932 |
| XGBoost             | ~19,418    | ~14,151   | 0.932 |
| Lasso               | ~19,835    | ~14,410   | 0.929 |
| Ridge               | ~20,190    | ~14,492   | 0.926 |
| Random Forest       | ~22,247    | ~15,656   | 0.910 |
| Linear Regression   | ~23,014    | ~15,903   | 0.904 |

*(exact values are reproduced in the notebook's Step 6 comparison table)*

## Video

📹 Video walkthrough: `https://drive.google.com/file/d/1iwpShuBQt7ILoohxbKNfZVbBNTKwYBUG/view?usp=drive_link`

## Author
 Smit Patel


Practical Exam submission — Red & White Skill Education, Supervised Learning, Set A.
