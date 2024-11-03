import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.model_selection import train_test_split

# Sample data for testing (replace with actual test data as needed)
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 5)
y_test = np.random.randint(0, 2, 20)

def test_train_model_type():
    """
    Test if the train_model function returns a RandomForestClassifier instance.
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier instance"

def test_compute_model_metrics_values():
    """
    Test if compute_model_metrics returns expected precision, recall, and fbeta values.
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_preds = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    
    # Manually computed values for the above example
    expected_precision = precision_score(y_true, y_preds)
    expected_recall = recall_score(y_true, y_preds)
    expected_fbeta = fbeta_score(y_true, y_preds, beta=1)
    
    assert precision == expected_precision, "Precision does not match expected value"
    assert recall == expected_recall, "Recall does not match expected value"
    assert fbeta == expected_fbeta, "F-beta does not match expected value"

def test_train_test_split_sizes():
    """
    Test if the train and test datasets have the expected size.
    """
    # Create a sample dataset with 100 samples and 5 features
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    # Perform an 80-20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the expected train and test sizes
    train_size = 80
    test_size = 20

    # Assert the sizes
    assert len(X_train) == train_size, "Training data size does not match expected size"
    assert len(X_test) == test_size, "Test data size does not match expected size"
    assert len(y_train) == train_size, "Training labels size does not match expected size"
    assert len(y_test) == test_size, "Test labels size does not match expected size"