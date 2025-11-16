import pandas as pd

def get_feature_names(preprocessor, numeric, categorical):
    """Extract OneHot-expanded feature names from ColumnTransformer."""
    
    numeric_features = numeric
    cat_features = preprocessor.named_transformers_["cat"] \
                                .named_steps["encoder"] \
                                .get_feature_names_out(categorical)

    return list(numeric_features) + list(cat_features)

def aggregate_importance(feature_names, importances):
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["original"] = df["feature"].apply(lambda x: x.split("_")[0])
    return df.groupby("original")["importance"].sum().sort_values(ascending=False)
