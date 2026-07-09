import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Demand Forecast Dashboard (Dark Mode)",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
    <style>
        .main {
            background-color: #0f172a;
            color: #e2e8f0;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #f1f5f9;
            font-weight: 700;
        }
        .metric-card {
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 16px;
            border: 1px solid #334155;
            box-shadow: 0 0 10px rgba(56,189,248,0.4);
            text-align: center;
        }
        .stPlotlyChart, .stMarkdown, .stDataFrame {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 0 6px rgba(56,189,248,0.3);
        }
        .stSidebar {
            background-color: #1e293b !important;
        }
    </style>
""", unsafe_allow_html=True)


# LOAD DATA

@st.cache_data
def load_all_data():
    actual_df = pd.read_parquet("daily_category_sum.parquet")
    backtest_df = pd.read_parquet("forecast_backtest_7days_sum.parquet")
    future_df = pd.read_parquet("forecast_future_7days_sum.parquet")
    return actual_df, backtest_df, future_df

actual_df, backtest_df, future_df = load_all_data()


# SIDEBAR

st.sidebar.title("Navigation")
category = st.sidebar.selectbox("Select Product Category", sorted(actual_df["l1_category"].unique()))
st.sidebar.info("Choose a category to compare backtest and forecast results.")



# DATA PREPARATION

cat_actual = actual_df[actual_df["l1_category"] == category].copy()
cat_backtest = backtest_df[backtest_df["l1_category"] == category].copy()
cat_future = future_df[future_df["l1_category"] == category].copy()

# Focus only on July 3–17
cat_actual = cat_actual[(cat_actual["date"] >= "2022-07-03") & (cat_actual["date"] <= "2022-07-10")]
cat_backtest = cat_backtest[(cat_backtest["date"] >= "2022-07-03") & (cat_backtest["date"] <= "2022-07-10")]
cat_future = cat_future[(cat_future["date"] >= "2022-07-11") & (cat_future["date"] <= "2022-07-17")]

st.title("📈 Future Demand Forecasting Dashboard")

st.markdown("---")

st.subheader(f"Category: {category}")

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 5))

# Add shaded regions for context
ax.axvspan(pd.Timestamp("2022-07-03"), pd.Timestamp("2022-07-10"), color="#1d4ed8", alpha=0.1, label="Backtest Period")
ax.axvspan(pd.Timestamp("2022-07-11"), pd.Timestamp("2022-07-17"), color="#16a34a", alpha=0.1, label="Forecast Period")

# Plot lines
ax.plot(cat_actual["date"], cat_actual["total_qty"], label="Actual (July 3–10)", color="#60a5fa", linewidth=2)
ax.plot(cat_backtest["date"], cat_backtest["forecast_qty"], label="Backtest Prediction (July 3–10)", 
        color="#fb923c", linestyle="--", linewidth=2)
ax.plot(cat_future["date"], cat_future["forecast_qty"], label="Future Forecast (July 11–17)", 
        color="#34d399", linestyle="--", linewidth=2)

# Chart details
ax.set_facecolor("#0f172a")
ax.set_title(f"{category} — Actual vs Predicted vs Future Forecast", fontsize=14, color="#f8fafc", weight="bold")
ax.set_xlabel("Date", color="#94a3b8")
ax.set_ylabel("Quantity Sold", color="#94a3b8")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=9)
st.pyplot(fig)


# METRICS

st.markdown("### Model Backtest Performance (July 3–10)")

if not cat_backtest.empty and not cat_actual.empty:
    # Ensure consistent column names before merge
    cat_actual = cat_actual.rename(columns={"total_qty": "actual_qty"})
    cat_backtest = cat_backtest.rename(columns={"forecast_qty": "pred_qty"})

    merged = pd.merge(cat_actual[["date", "actual_qty"]],
                      cat_backtest[["date", "pred_qty"]],
                      on="date", how="inner")

    if len(merged) > 0:
        rmse = mean_squared_error(merged["actual_qty"], merged["pred_qty"], squared=False)
        mae = mean_absolute_error(merged["actual_qty"], merged["pred_qty"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>RMSE</h3><h2 style='color:#38bdf8'>{rmse:,.2f}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>MAE</h3><h2 style='color:#38bdf8'>{mae:,.2f}</h2></div>", unsafe_allow_html=True)
    else:
        st.warning("No overlapping dates found for backtest comparison.")
else:
    st.warning("Not enough data to compute metrics for this category.")


# INSIGHTS

st.markdown("### Insights & Interpretation")
st.markdown("""
- **Actual Sales (3–10 July)** — observed data used to validate the model.  
- **Backtest Prediction** — model performance on known data (should closely follow blue line).  
- **Future Forecast (11–17 July)** — projected demand ahead.  
- Shaded zones visually separate **evaluation period** (blue) and **forecast window** (green).  
""")

st.caption("Powered by LightGBM + Streamlit")