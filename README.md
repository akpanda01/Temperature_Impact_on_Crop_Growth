# 🌾 Temperature Impact on Crop Growth

This project uses machine learning to analyze how temperature influences crop yields. It involves data preprocessing, training a predictive model, and generating forecasts based on real agricultural datasets.

---

## 📁 Project Structure

```
TemperatureImpactOnCropGrowth/
│
├── data/                  # Dataset (samples or full data if small)
├── models/                # Trained model files
├── docs/                  # Report, README, presentations
├── notebooks/             # Optional notebooks for EDA
└── requirements.txt       # Python dependencies
```

---

## 🛠️ Setup Instructions

1. **Clone the Repository / Extract ZIP:**

```
git clone <your_repo_url>
# OR unzip the provided ZIP file
```

2. **Install Dependencies:**

Make sure Python ≥ 3.8 is installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

- If the dataset is large, only include a sample (`sample_train.csv`, `sample_test.csv`) inside the `data/` folder.
- For full training, download and place the CSV in `data/` as `yield_df.csv`.

---

## 🧪 Training the Model

Run the training script to build the model and save it:

```bash
python src/train.py
```

## 📈 Performance

The model aims to achieve **>80% R² score** (or mAP if classification) using RandomForest or XGBoost regressors.

---

## 📚 Documentation

See `docs/Project methodology.pdf`

---

## 🤖 Author

- Name: Akanksha Panda
- Project: Machine Learning-based Crop Yield Prediction
- Submitted for: CDS3005- Foundations of Data Science
