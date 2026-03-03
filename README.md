#  Crop Yield Predictor

A machine learning web app that predicts crop yield based on country, climate, and crop type. Built with Python, Scikit-learn, and Streamlit.

---

## Demo

Input your crop, country, rainfall, and temperature — get a predicted yield in kg/ha and tonnes/ha, scaled to your field size.

---

## Dataset

**Source:** [Agriculture Crop Yield – Kaggle](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield)


**Features:** `Area`, `Item`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`  
**Target:** `hg/ha_yield` (hectograms per hectare)

---

## Model

Three models were tested on an 80/20 train-test split:

| Model | R² | MAE |
|---|---|---|
| Linear Regression | 0.835 | — |
| Gradient Boosting | 0.824 | — |
| **Random Forest** | **0.983** | **— (best)** |

**Random Forest** was selected for its ability to capture non-linear relationships between climate variables and yield.

### Per-Crop Models

Rather than one global model, a dedicated Random Forest model is trained per crop. This lets each model specialise — maize in Brazil behaves very differently to maize in Canada.

| Crop | Rows | R² | MAE % | CV R² Mean | CV R² Std |
|---|---|---|---|---|---|
| Cassava | 2045 | 0.962 | 5.6% | 0.955 | 0.013 |
| Wheat | 3857 | 0.957 | 7.3% | 0.961 | 0.005 |
| Rice, paddy | 3388 | 0.944 | 6.2% | 0.934 | 0.014 |
| Potatoes | 4276 | 0.943 | 6.5% | 0.941 | 0.004 |
| Yams | 847 | 0.939 | 5.3% | 0.939 | 0.014 |
| Maize | 4121 | 0.918 | 10.9% | 0.929 | 0.010 |
| Soybeans | 3223 | 0.897 | 8.0% | 0.901 | 0.009 |
| Sweet potatoes | 2890 | 0.893 | 6.9% | 0.923 | 0.024 |
| Sorghum | 3039 | 0.877 | 12.1% | 0.868 | 0.027 |
| Plantains and others | 556 | 0.714 | 14.5% | 0.807 | 0.074 |

Results were validated with **5-fold cross validation**. Test R² and CV R² are consistent across all crops, confirming the models are genuinely learning patterns and not overfitting to a lucky split.

> **Note on Plantains:** Lowest performer due to only 556 training samples — less than a third of most other crops. This is a data limitation, not a model failure.

### Preprocessing

- `Year` and `Unnamed: 0` columns dropped
- `Area` and `Item` one-hot encoded (no ordinal relationship between countries/crops)
- `pesticides_tonnes` log-transformed with `np.log1p()` to handle extreme right skew (range: 0.04 – 367,778 tonnes)
- Features scaled with `StandardScaler`
- Pesticide input is auto-filled from country historical average at prediction time — farmers don't need to know national pesticide tonnage

---

## Project Structure

```
crop_yield_predictor/
├── jupyter/
│   └── build_models.ipynb         ← data exploration, model training, export
|   └── models/
│   ├── meta.json           ← crop metadata, valid countries, input ranges
│   ├── Maize_model.pkl
│   ├── Maize_scaler.pkl
│   └── ...                 ← one model + scaler per crop
├── app.py                  ← Streamlit web app

└── README.md
```


## Limitations

- Dataset is global and historical (1990–2013) — not hyperlocal or real-time
- Pesticide data is national aggregate, not per-farm
- No soil type, irrigation, or field-size data in the training set
- `hg/ha_yield` is a rate — multiply by field size to get total expected harvest
