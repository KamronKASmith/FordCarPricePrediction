import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', LinearRegression())
    ])
    return pipeline

def train_model(df: pd.DataFrame):
    X = df.drop(columns='price')
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n📊 Linear Regression Performance:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    return pipeline
