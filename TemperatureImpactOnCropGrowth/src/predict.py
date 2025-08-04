import joblib
import numpy as np

MODEL_PATH = "crop_model.pkl"


def predict_crop(input_features):
    model, le = joblib.load(MODEL_PATH)
    input_array = np.array(input_features).reshape(1, -1)
    pred = model.predict(input_array)
    return le.inverse_transform(pred)[0]


if __name__ == "__main__":
    # Example input [N, P, K, temperature, humidity, ph, rainfall]
    sample_input = [90, 42, 43, 20.8797, 82.0027, 6.502985, 202.9355]
    prediction = predict_crop(sample_input)
    print(f"Predicted Crop: {prediction}")
