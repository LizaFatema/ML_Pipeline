import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# ── Step 1: Load dataset ──────────────────────────────────────────────────────

df = pd.read_csv("insurance.csv")
print(df)
# except FileNotFoundError:
#     raise FileNotFoundError(
#         "insurance.csv not found in the current folder.\n"
#         "Download it from: https://www.kaggle.com/datasets/mirichoi0218/insurance\n"
#         "and place it in the same folder as rf_train.py"
#     )
 
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()
 
# ── Step 2: Define features ───────────────────────────────────────────────────
print("=" * 55)
print("STEP 2: Defining feature types")
print("=" * 55)
 
numeric_features     = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']
target_col           = 'expenses'
 
# Handle both column name variants from the dataset
if 'charges' in df.columns and 'expenses' not in df.columns:
    df.rename(columns={'charges': 'expenses'}, inplace=True)
    print("Note: renamed 'charges' -> 'expenses'")
 
print(f"Numeric features    : {numeric_features}")
print(f"Categorical features: {categorical_features}")
print(f"Target              : {target_col}")
print()
 
# ── Step 3: Split ─────────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 3: Train / Test split (80 / 20)")
print("=" * 55)
 
X = df[numeric_features + categorical_features]
y = df[target_col]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows : {X_train.shape[0]}")
print(f"Test rows     : {X_test.shape[0]}")
print()
 
# ── Step 4: Build preprocessing pipeline ──────────────────────────────────────
print("=" * 55)
print("STEP 4: Building preprocessing pipeline")
print("=" * 55)
 
# Numeric: impute missing values with median, then standardise
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
 
# Categorical: impute with most frequent, then one-hot encode
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
 
# ColumnTransformer applies each sub-pipeline to the right columns
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])
 
print("Preprocessor built.")
print(f"  Numeric    -> SimpleImputer + StandardScaler  ({numeric_features})")
print(f"  Categorical-> SimpleImputer + OneHotEncoder   ({categorical_features})")
print()
 
# ── Step 5: Full pipeline with GradientBoostingRegressor ──────────────────────
print("=" * 55)
print("STEP 5: Building full pipeline (preprocessor + model)")
print("=" * 55)
 
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])
 
print("Full pipeline structure:")
print(pipeline)
print()
 
# ── Step 6: GridSearchCV to find best hyperparameters ─────────────────────────
print("=" * 55)
print("STEP 6: GridSearchCV (this may take 1-2 minutes)")
print("=" * 55)
 
param_grid = {
    'model__n_estimators' : [50, 100, 200],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth'    : [2, 3, 5],
}
 
print("Parameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")
print(f"Total combinations: {3*3*3} (each tested with 5-fold CV)")
print()
 
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
 
print()
print("Best Parameters Found:")
for param, val in grid_search.best_params_.items():
    print(f"  {param}: {val}")
print(f"Best CV R2: {grid_search.best_score_:.4f}")
print()
 
# ── Step 7: Evaluate best model on test set ───────────────────────────────────
print("=" * 55)
print("STEP 7: Evaluating best model on test set")
print("=" * 55)
 
best_model  = grid_search.best_estimator_
y_pred      = best_model.predict(X_test)
mae         = mean_absolute_error(y_test, y_pred)
rmse        = np.sqrt(mean_squared_error(y_test, y_pred))
r2          = r2_score(y_test, y_pred)
 
print(f"MAE      : ${mae:,.2f}")
print(f"RMSE     : ${rmse:,.2f}")
print(f"R2 Score : {r2:.4f}")
print()
 
# ── Step 8: Save the pipeline as .pkl ─────────────────────────────────────────
print("=" * 55)
print("STEP 8: Saving pipeline to best_model_pipeline.pkl")
print("=" * 55)
 
with open("best_model_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("best_model_pipeline.pkl saved.")
 
# Save feature metadata (used by app.py for input validation)
feature_info = {
    'numeric_features'    : numeric_features,
    'categorical_features': categorical_features,
    'target_col'          : target_col,
    'sex_options'         : sorted(df['sex'].unique().tolist()),
    'smoker_options'      : sorted(df['smoker'].unique().tolist()),
    'region_options'      : sorted(df['region'].unique().tolist()),
    'age_min'             : int(df['age'].min()),
    'age_max'             : int(df['age'].max()),
    'bmi_min'             : float(df['bmi'].min()),
    'bmi_max'             : float(df['bmi'].max()),
    'children_max'        : int(df['children'].max()),
}
with open("feature_info.pkl", "wb") as f:
    pickle.dump(feature_info, f)
print("feature_info.pkl saved.")
 
# ── Step 9: Quick reload verification ─────────────────────────────────────────
print()
print("=" * 55)
print("STEP 9: Reload verification")
print("=" * 55)
 
with open("best_model_pipeline.pkl", "rb") as f:
    loaded = pickle.load(f)
 
r2_check = r2_score(y_test, loaded.predict(X_test))
print(f"Reload check - Test R2: {r2_check:.4f}")
print()
print("=" * 55)
print("ALL DONE!")
print("=" * 55)
print()
print("Files created in this folder:")
print("  best_model_pipeline.pkl   <- trained pipeline (use in app.py)")
print("  feature_info.pkl          <- feature metadata (optional, for validation)")
print()
print("Next steps:")
print("  1. Keep both .pkl files in the same folder as app.py")
print("  2. Run:  python app.py")
print("  3. Open: http://127.0.0.1:7860  in your browser")