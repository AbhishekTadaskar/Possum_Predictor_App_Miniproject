import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load the dataset
df = pd.read_csv("possum.csv")

# 1. Data Cleaning
df_cleaned = df.drop(columns=['case', 'site'])
numerical_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(df_cleaned[numerical_cols].mean())

df_cleaned['sex'] = df_cleaned['sex'].map({'m': 0, 'f': 1})
df_cleaned = pd.get_dummies(df_cleaned, columns=['Pop'], drop_first=True)

# 2. Define features (X) and target (y)
target_column = 'totlngth'
y = df_cleaned[target_column]

# Define the exact order of features
feature_columns = ['sex', 'age', 'hdlngth', 'skullw', 'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly', 'Pop_other']
X = df_cleaned[feature_columns]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Set up GridSearchCV with K-Fold cross-validation
param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=kf, verbose=1, n_jobs=-1)

# 5. Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# 6. Get the best model
best_model = grid_search.best_estimator_

# 7. Evaluate and save the best model
predictions = best_model.predict(X_test)
final_r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Final R2 score on the test set: {final_r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Save the best model
file_path = "best_possum_model.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(best_model, file)
print(f"\nModel successfully saved to {file_path}")