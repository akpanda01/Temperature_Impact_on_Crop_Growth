from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import json

def create_model():
    return LogisticRegression(max_iter=1000)

def evaluate_model(model, X_test, y_test, label_encoder, metrics_path="metrics.json"):
    y_pred = model.predict(X_test)

    # Generate report
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=4)

    return report, cm
