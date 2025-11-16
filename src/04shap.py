import numpy as np
import shap

def compute_shap_values(best_model, X_train, preprocessor):
    """Compute SHAP values for RandomForest/XGBoost inside pipeline."""
    
    model = best_model.named_steps["model"]
    X_transformed = preprocessor.transform(X_train)

    # Convert sparse matrix to dense if needed
    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray().astype(float)
    else:
        X_dense = X_transformed.astype(float)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dense)

    return shap_values, X_dense
