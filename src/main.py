from data_cleaning import clean_dataset
from feature_engineering import build_preprocessor
from modeling import train_random_forest
from utils import get_feature_names, aggregate_importance

import pandas as pd

def main():

    df = clean_dataset("data/raw/house_data.xlsx")

    numeric_features = ["area", "rooms", "halls", "total_floor"]
    categorical_features = ["orientation","decoration","floor_level","building_type","subdistrict"]

    X = df[numeric_features + categorical_features]
    y = df["price"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print("Training RandomForest...")
    rf_grid = train_random_forest(preprocessor, X_train, y_train)

    print("Best params:", rf_grid.best_params_)

    model = rf_grid.best_estimator_
    feature_names = get_feature_names(
        model.named_steps["preprocessor"],
        numeric_features,
        categorical_features
    )
    importances = model.named_steps["model"].feature_importances_

    agg = aggregate_importance(feature_names, importances)
    print("Aggregated Feature Importance:")
    print(agg)

if __name__ == "__main__":
    main()
