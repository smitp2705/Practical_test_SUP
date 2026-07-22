# Summary Report — House Price Prediction (Supervised Learning)

## Business Problem & Dataset

A PropTech startup (in the spirit of NoBroker/MagicBricks) wants an automated house price
estimator to help buyers and sellers benchmark property values. We used the **Ames Housing
Dataset** (`train.csv`, 1,460 rows, 79 descriptive features + `Id` and `SalePrice`), a well-known
regression benchmark that captures nearly every attribute of a residential home — size, quality,
location, age, amenities, and sale conditions. The task is framed as **supervised regression**:
predict the continuous `SalePrice` from the property's features.

## Preprocessing & Feature Engineering

Missing values were handled based on *why* they were missing: for columns like `PoolQC`,
`GarageType`, and `BsmtQual`, a NaN means the house genuinely lacks that feature, so these were
filled with `'None'`. Numeric companions (`GarageArea`, `BsmtFinSF1`, etc.) were filled with `0`
for the same reason. Remaining numeric gaps were filled with the column median, and no column
exceeded the 80% missing threshold that would have warranted dropping it outright. Two extreme
outliers (`GrLivArea` > 4,000 sq ft but `SalePrice` < $300,000) were removed, since a couple of
contradictory high-leverage points can distort a linear model's fitted coefficients for the
remaining 99% of "normal" houses.

We engineered `TotalSF` (basement + 1st + 2nd floor area, a single strong size signal),
`HouseAge` and `RemodAge` (capturing depreciation/renovation recency), and binary `HasGarage` /
`HasPool` flags. Quality-related ordinal columns (`ExterQual`, `KitchenQual`, `BsmtQual`,
`FireplaceQu`) were mapped to a 0–5 ordinal scale rather than one-hot encoded, since they have a
natural ordering. Low-cardinality nominal categoricals were one-hot encoded, while the
high-cardinality `Neighborhood` column was **mean target encoded** (fit only on the training
split, to avoid leakage) rather than one-hot encoded, since one-hot would have added 25 sparse
columns. Skewed numeric features (skew > 0.75) were log1p-transformed before scaling, and the
target itself (`SalePrice`) was log1p-transformed prior to the train/test split, since raw sale
prices are right-skewed and violate the linear-regression assumption of normally distributed
errors. All of this was wrapped in a single `ColumnTransformer` + `Pipeline` for reproducibility.

## Best Model & Why

Five models were compared: Linear Regression, Ridge, Lasso, Random Forest, and XGBoost (plus a
hyperparameter-tuned XGBoost via `RandomizedSearchCV`). The **tuned XGBoost Regressor** performed
best, with the lowest test RMSE (~$19,332) and highest R² (0.932), edging out the untuned
XGBoost, Lasso, Ridge, Random Forest, and Linear Regression in that order. XGBoost's gradient-
boosted trees naturally capture non-linear relationships and feature interactions (e.g., how
`OverallQual` and `TotalSF` jointly affect price) that linear models can only approximate through
manual interaction terms. Its main trade-off is reduced interpretability versus Ridge/Lasso,
which we addressed via feature importance rankings and plain-English business interpretation.

## Top 3 Most Predictive Features

1. **OverallQual** — overall material/finish quality is the single strongest price driver;
   real estate professionals should treat renovation/quality upgrades as a high-ROI lever.
2. **TotalSF** (engineered total living area) — confirms the intuitive "price per square foot"
   heuristic agents already use, but as a data-driven, combined metric rather than a single floor.
3. **KitchenQual** — kitchen condition is disproportionately influential relative to its physical
   footprint, reinforcing the common renovation advice to prioritize kitchen upgrades before sale.

## Next Steps for Production

- **More data**: incorporate additional years/markets beyond Ames to improve generalization and
  reduce the wider error band seen for luxury/starter homes in the residual analysis.
- **Ensembling**: blend/stack XGBoost with Ridge/Lasso predictions, since linear models sometimes
  generalize better at the distribution's extremes.
- **Bayesian optimization** (e.g., Optuna) in place of `RandomizedSearchCV` for a more efficient,
  targeted hyperparameter search.
- **Monitoring & retraining**: track prediction drift as market conditions change, and retrain
  periodically with fresh listings.
- **SHAP-based explanations** in the product UI, so agents/buyers see *why* a given estimate was
  produced (already sketched as an optional bonus in the notebook).
