# Pneumonia Detection from Chest X-Rays (Weighted Ensemble)

A 5-model weighted ensemble (EfficientNetV2L, EfficientNetB7, InceptionResNetV2, DenseNet201, NASNetLarge) for pneumonia detection.

## Setup Instructions

1. Clone repository and create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Mendeley Chest X-Ray dataset and place it in `data/chest_xray/`.

## Execution

* **To Train:** `python scripts/train_all.py` (Requires Google Colab T4)
* **To Evaluate:** `python scripts/evaluate_ensemble.py`
* **To Run Web App:** `streamlit run app.py`