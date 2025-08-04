from data_preprocessing import load_and_preprocess_data
from model_utils import create_model, evaluate_model
import joblib

LOG_PATH = "training_log.txt"
MODEL_PATH = "crop_model.pkl"
CSV_PATH = "Crop_recommendation.csv"


def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def main():
    log("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data(CSV_PATH)

    log("Creating model...")
    model = create_model()

    log("Training model...")
    model.fit(X_train, y_train)

    log("Evaluating model...")
    report, cm = evaluate_model(model, X_test, y_test, le)

    log("Saving model...")
    joblib.dump((model, le), MODEL_PATH)

    log("Training complete.")


if __name__ == "__main__":
    main()
