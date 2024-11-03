import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
def main():
    # Define paths
    project_path = (
        "/mnt/c/Users/kaleb/Desktop/DEPLOYING-A-SCALABLE-ML-PIPELINE-WITH-FASTAPI"
    )
    data_path = os.path.join(project_path, "data", "census.csv")
    model_dir = os.path.join(project_path, "model")
    os.makedirs(model_dir, exist_ok=True)  
    print(f"Data path: {data_path}")
    data = pd.read_csv(data_path)
    # Split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # DO NOT MODIFY
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Train and save the model
    model = train_model(X_train, y_train)
    model_path = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    save_model(model, model_path)
    save_model(encoder, encoder_path)
    # Load the model
    model = load_model(model_path)
    # Inference and metrics
    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")
    # Compute performance on categorical slices
    with open("slice_output.txt", "w") as f:  
        for col in cat_features:
            for slice_value in sorted(test[col].unique()):
                count = test[test[col] == slice_value].shape[0]
                p, r, fb = performance_on_categorical_slice(
                    data=test,
                    column_name=col,
                    slice_value=slice_value,
                    categorical_features=cat_features,
                    label="salary",
                    encoder=encoder,
                    lb=lb,
                    model=model,
                )
                f.write(f"{col}: {slice_value}, Count: {count:,}\n")
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
if __name__ == "__main__":
    main()
