# 🚗 Car Price Prediction & Business Intelligence Dashboard

> **An end-to-end machine learning and business intelligence web application** built with Streamlit, scikit-learn, and Plotly.  
> Predict used car prices in the Indian market and explore rich BI analytics — all in a single Python file.

---

## 📦 Dataset

| Field | Detail |
|---|---|
| **Source** | [Vehicle Dataset from CarDekho — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |
| **Rows** | ~8,128 listings |
| **Features** | `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `seats` |
| **Target** | `selling_price` (INR) |

---

## 🚀 Quick Start

### 1. Clone / download the repo

```bash
git clone https://github.com/<your-username>/car-price-bi.git
cd car-price-bi
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

> **Note:** Place `car price prediction.csv` in the same directory as `app.py`.

---

## 🗂️ Project Structure

```
car-price-bi/
│
├── app.py                        # ← Single-file Streamlit application
├── car price prediction.csv      # ← Dataset (download from Kaggle link above)
├── requirements.txt              # ← Python dependencies
├── README.md                     # ← This file
└── Car_Price_Prediction_Report.docx  # ← Structured project report
```

---

## 📊 Application Tabs

| Tab | Contents |
|---|---|
| **Exploratory Analysis** | Price distribution, fuel-type boxplots, brand rankings, year trend, KM-vs-price scatter, correlation heatmap |
| **ML Models & Accuracy** | Train/test comparison of Gradient Boosting, Random Forest, Ridge Regression; R², MAE, RMSE; feature importances; residuals |
| **Price Predictor** | Interactive form → instant price estimate + ±15% market range + similar listings |
| **Business Intelligence** | Market share, depreciation curve, brand volume/price scatter, seller-type analysis, fuel×transmission sunburst |
| **Raw Data** | Filtered dataset preview with CSV download and descriptive statistics |

---

## 🤖 Machine Learning Pipeline

```
Raw CSV
  │
  ▼
Data Cleaning (null imputation, type casting)
  │
  ▼
Feature Engineering
  • car_age  = 2024 − year
  • brand    = first token of car name
  • LabelEncoding for categorical features
  │
  ▼
Train / Test Split (80 / 20, random_state=42)
  │
  ▼
Three Models trained in parallel:
  • GradientBoostingRegressor  ← typically best R²
  • RandomForestRegressor
  • Ridge Regression
  │
  ▼
Best model selected by R² → used for live predictions
```

### Typical Performance

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Gradient Boosting | ~0.88 | ~₹62,000 | ~₹1,12,000 |
| Random Forest | ~0.86 | ~₹68,000 | ~₹1,20,000 |
| Ridge Regression | ~0.72 | ~₹1,10,000 | ~₹1,80,000 |

> Actual numbers depend on the random seed and data version.

---

## 📌 Key Business KPIs

| KPI | Description |
|---|---|
| **Average Selling Price** | Central price benchmark across inventory |
| **Median Price** | Robust price anchor; less affected by premium outliers |
| **Price by Fuel Type** | Diesel > Petrol > CNG > LPG in resale value |
| **Depreciation Rate** | ~15–20% per year in first 5 years |
| **Automatic Premium** | Automatics command 35–55% premium over manuals |
| **Dealer Certification Uplift** | Dealer listings average 20–40% above individual listings |

---

## ⚠️ Key Risks

1. **Depreciation cliff** — vehicles older than 10 years cluster below ₹3 Lakh.
2. **Fuel-type obsolescence** — LPG/CNG resale demand is declining.
3. **Over-reliance on single features** — max_power and engine cc dominate; data bias toward Maruti/Hyundai.
4. **Model drift** — price predictions will need retraining as market conditions shift.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI / App | [Streamlit](https://streamlit.io) |
| Data Manipulation | [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| ML | [scikit-learn](https://scikit-learn.org) |
| Visualisation | [Plotly Express](https://plotly.com/python/plotly-express/) |
| Report Generation | [python-docx](https://python-docx.readthedocs.io) |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built as part of an IBM Internship Data Science project.*
