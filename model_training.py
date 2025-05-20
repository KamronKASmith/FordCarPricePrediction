import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Pipeline hosts Models
def build_pipeline(X: pd.DataFrame, model_type: str = "linear") -> Pipeline:
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    #Selecting the model
    if model_type == "random":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "boost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == "light":
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1) #Verbose Line to remove LightGBM Info statements
    elif model_type == "ridge":
        model = Ridge(alpha=1.0, max_iter=10000)
    elif model_type == "lasso":
        model = Lasso(alpha=0.1, max_iter=10000)
    elif model_type == "tree":
        model = DecisionTreeRegressor(max_depth=5)
    else:
        model = LinearRegression()

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    return pipeline

#Trains the model selected with hyperparameters
def train_model(df, model_type):
    X = df.drop(columns='price')
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(X, model_type)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        'pipeline': pipeline,
        'y_test': y_test,
        'y_pred': y_pred
    }
