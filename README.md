# 📈 Future Demand Forecasting System

A machine learning-based demand forecasting system developed using **LightGBM** and **Streamlit** to predict future product demand across multiple product categories using historical sales data. The project leverages **46.7+ million Flipkart transaction records** (April–July 2022) and provides an interactive dashboard for comparing historical demand, model backtesting, and future demand forecasts to support inventory planning and decision-making.

> This project was developed as an academic project at the **National Institute of Technology Calicut, Department of Computer Science**, under the guidance of **Vidhya Kamakshi** and **Pournami P.N.** See [`ML_Project_Report.pdf`](./ML_Project_Report.pdf) for the full methodology and write-up.

---

## 🚀 Features

- Forecasts future demand using **per-category LightGBM regression models**
- Covers **189 usable product categories** (out of 200+ in the raw data)
- Interactive **Streamlit dashboard** with a dark-themed UI
- Actual vs. Backtest vs. Future forecast visualization per category
- 7-day future demand forecasting via a recursive, day-by-day forecasting approach
- Model performance evaluation using **RMSE** and **MAE**, computed per category

---

## 🛠️ Tech Stack

**Dashboard app (`requirements.txt`):**
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyArrow

**Model training (used in the notebooks, not in `requirements.txt`):**
- LightGBM — core forecasting model
- Optuna — Bayesian hyperparameter optimization
- Seaborn, tqdm, psutil — visualization, progress tracking, memory management

> ℹ️ If you plan to re-run the training notebooks, install these extra packages separately (e.g. `pip install lightgbm optuna seaborn tqdm psutil`).

---

## 📂 Project Structure

```
Future-Demand-Forecasting-System/
│
├── app.py
├── baseline_model.ipynb
├── lightgbm_enhanced_training.ipynb
├── daily_category_sum.parquet
├── forecast_backtest_7days_sum.parquet
├── forecast_future_7days_sum.parquet
├── requirements.txt
├── ML_Project_Report.pdf
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses a **Flipkart grocery transaction dataset** covering **April 2022 to July 2022**, containing **46,706,387 cleaned transaction records** across **200+ L1 product categories** (e.g. Milk, Fresh Vegetables, Bread & Pav, Curd & Yogurt).

Due to GitHub file size limitations, the original raw dataset is **not included** in this repository — only the aggregated/processed Parquet files needed to run the dashboard are provided.

**Preprocessing summary:**
- Removed irrelevant fields (product IDs, SKU codes, brand names, delivery slots) and duplicate transactions
- Handled missing/invalid categories, negative quantities, and missing prices (imputed via category median)
- Standardized all timestamps to `YYYY-MM-DD`
- Aggregated to **daily total quantity** and **average price** per category
- Filtered out low-activity categories with fewer than 20 valid daily records (e.g. Free Store, Freebie, Combo & Recipes, Smoking Cessation)
- Final cleaned dataset: **189 usable categories**, continuous daily series with no missing target values

---

## 🧠 How It Works

**1. Feature Engineering**
For each category, the following predictors were engineered:
- **Lag features:** `lag_1`, `lag_7`, `lag_14`
- **Rolling statistics:** `r7_mean`, `r14_mean`, `r7_std`
- **Price feature:** `avg_price` (category-level daily mean, smoothed for missing values)
- **Calendar features:** `day_of_week`, `month`

The most influential features by importance were `lag_1`, `r14_mean`, `lag_14`, `r7_std`, and `avg_price`.

**2. Hyperparameter Tuning**
Hyperparameters were tuned with **Optuna**, using a time-based split (train on data before July 3, validate on July 3–10) to avoid data leakage. The best configuration found:

```python
final_params = {
    'learning_rate': 0.14968095204141948,
    'num_leaves': 104,
    'feature_fraction': 0.907180607741709,
    'bagging_fraction': 0.8916015716223886,
    'bagging_freq': 6,
    'lambda_l1': 0.7921484634069514,
    'lambda_l2': 1.4607774927730666,
    'min_data_in_leaf': 83,
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
}
```

LightGBM's early stopping (`stopping_rounds=50`) was used to prevent overfitting.

**3. Backtesting (July 3–10, 2022)**
A separate model was trained per category and evaluated against actual sales for the most recent 7 known days.

**4. Future Forecasting (July 11–17, 2022)**
Forecasts were generated using a **recursive, day-by-day approach**: each day's features (lags, rolling stats, calendar values) are recomputed from the latest known/predicted values, and the model predicts one day at a time, feeding each prediction back into the series — simulating a real production forecasting pipeline.

**5. Dashboard (`app.py`)**
Loads the three processed Parquet files, lets the user pick a category, and plots **actual vs. backtest vs. future forecast** on a single dark-themed chart with shaded backtest/forecast regions, plus RMSE/MAE metric cards for the selected category.

---

## 📉 Model Performance (Backtest, July 3–10, 2022)

Evaluated across all 189 usable categories:

| Metric | Value |
|---|---|
| Average RMSE | 303.17 |
| Median RMSE | 134.53 |
| Average MAE | 246.95 |
| Best-performing category | Mother Care (RMSE = 1.34) |
| Worst-performing category | Fresh Vegetables (RMSE = 3589.99) |
| Forecast Horizon | 7 days |
| Model | LightGBM (per-category) |

The model performs strongly on stable, slow-moving categories, while highly volatile/perishable categories (e.g. Fresh Vegetables, Fresh Fruits) show higher error — consistent with their inherently noisier demand patterns.

---

## ⚙️ Installation

Clone the repository
```bash
git clone https://github.com/<your-username>/Future-Demand-Forecasting-System.git
```

Navigate to the project
```bash
cd Future-Demand-Forecasting-System
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run the Streamlit application
```bash
streamlit run app.py
```

> ⚠️ The date ranges in `app.py` are currently hardcoded to match this dataset's backtest (July 3–10, 2022) and forecast (July 11–17, 2022) windows. Update them if you retrain on new data.

---

## 📌 Future Improvements

- Real-time demand prediction using continuously retrained models
- Multi-city / multi-warehouse forecasting
- Further hyperparameter optimization and ensembling
- Cloud deployment with Docker and AWS

---

## 👨‍💻 Author

**Kevalsinh Kumpavat**

Department of Computer Science, National Institute of Technology Calicut
Under the guidance of **Vidhya Kamakshi** and **Pournami P.N.**

GitHub: https://github.com/thunder010

---

## 📄 License

This project is intended for educational and research purposes.