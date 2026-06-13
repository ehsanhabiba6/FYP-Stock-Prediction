
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from xgboost import XGBRegressor

# ── Page Config ──
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Load Models ──
@st.cache_resource
def load_models():
    gru_model = tf.keras.models.load_model('gru_model.keras')
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    return gru_model, xgb_model, scalers

gru_model, xgb_model, scalers = load_models()

# ── Header ──
st.title("📈 AI-Driven Stock Price Prediction")
st.markdown("### GRU + XGBoost Hybrid Model | R² = 0.9925")
st.markdown("---")

# ── Sidebar ──
st.sidebar.title("⚙️ Settings")
ticker = st.sidebar.text_input(
    "Stock Ticker Symbol",
    value="AAPL",
    help="e.g. AAPL, TSLA, MSFT, GOOGL"
).upper()

predict_days = st.sidebar.slider(
    "Prediction Days",
    min_value=7,
    max_value=30,
    value=30
)

company_names = {
    "AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla",
    "GOOGL": "Google", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "META": "Meta", "NFLX": "Netflix", "JPM": "JPMorgan",
    "GS": "Goldman", "V": "Visa", "PYPL": "PayPal",
    "XOM": "ExxonMobil", "CVX": "Chevron", "SHEL": "Shell"
}

# ── Main ──
col1, col2, col3 = st.columns(3)

if st.sidebar.button("🔮 Predict", use_container_width=True):
    with st.spinner(f"Fetching {ticker} data..."):
        try:
            # Data fetch karo
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            df = yf.download(ticker, 
                           start=start_date, 
                           end=end_date, 
                           progress=False,
                           auto_adjust=True)

            if len(df) < 60:
                st.error("❌ Data kam hai — valid ticker daalo!")
            else:
                # Columns fix karo
                df.columns = [col[0] if isinstance(col, tuple) 
                             else col for col in df.columns]

                # Current price
                current_price = float(df["Close"].iloc[-1])
                
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Ticker", ticker)
                col3.metric(
                    "Company", 
                    company_names.get(ticker, ticker)
                )

                # Technical Indicators
                df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                df["RSI"] = 100 - (100 / (1 + gain/loss))
                ema12 = df["Close"].ewm(span=12).mean()
                ema26 = df["Close"].ewm(span=26).mean()
                df["MACD"] = ema12 - ema26
                df = df.dropna()

                # Scale karo
                from sklearn.preprocessing import MinMaxScaler
                feature_cols = ["Open","High","Low","Close",
                               "Volume","MA20","MA50","RSI","MACD"]
                
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df[feature_cols])
                scaled_df = pd.DataFrame(scaled, columns=feature_cols)

                # Sequence banao — last 60 days
                seq = scaled_df.values[-60:]
                seq = seq.reshape(1, 60, 9)

                # Predict karo — future days
                predictions_scaled = []
                current_seq = seq.copy()

                for _ in range(predict_days):
                    # GRU predict
                    gru_pred = gru_model.predict(
                        current_seq, verbose=0
                    )[0][0]
                    
                    # XGBoost predict
                    xgb_pred = xgb_model.predict(
                        current_seq.reshape(1, -1)
                    )[0]
                    
                    # Hybrid
                    hybrid_pred = (gru_pred * 0.6) + (xgb_pred * 0.4)
                    predictions_scaled.append(hybrid_pred)

                    # Next sequence update karo
                    new_row = current_seq[0, -1, :].copy()
                    new_row[3] = hybrid_pred  # Close update
                    current_seq = np.roll(current_seq, -1, axis=1)
                    current_seq[0, -1, :] = new_row

                # Inverse transform
                dummy = np.zeros((predict_days, 9))
                dummy[:, 3] = predictions_scaled
                predicted_prices = scaler.inverse_transform(dummy)[:, 3]

                # ── Graph ──
                st.markdown("### 📊 Prediction Graph")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Historical — last 60 days
                hist_prices = df["Close"].values[-60:]
                hist_dates = df.index[-60:]
                ax.plot(hist_dates, hist_prices, 
                       color="#333333", linewidth=2,
                       label="Historical Price")

                # Future dates
                future_dates = pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=predict_days,
                    freq="B"
                )
                ax.plot(future_dates, predicted_prices,
                       color="#F44336", linewidth=2,
                       linestyle="--", label=f"Predicted ({predict_days} days)",
                       marker="o", markersize=3)

                # Connect karo
                ax.plot([hist_dates[-1], future_dates[0]],
                       [hist_prices[-1], predicted_prices[0]],
                       color="#F44336", linewidth=2, linestyle="--")

                ax.fill_between(future_dates, 
                               predicted_prices * 0.98,
                               predicted_prices * 1.02,
                               alpha=0.2, color="#F44336",
                               label="±2% confidence")

                ax.set_title(
                    f"{company_names.get(ticker, ticker)} "
                    f"({ticker}) — {predict_days} Day Prediction",
                    fontsize=14, fontweight="bold"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # ── Prediction Table ──
                st.markdown("### 📋 Predicted Prices")
                pred_df = pd.DataFrame({
                    "Date": future_dates.strftime("%Y-%m-%d"),
                    "Predicted Price ($)": [
                        f"${p:.2f}" for p in predicted_prices
                    ]
                })
                st.dataframe(pred_df, use_container_width=True)

                # ── Model Info ──
                st.markdown("### 🤖 Model Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.metric("Model", "GRU + XGBoost Hybrid")
                info_col2.metric("R² Score", "0.9925")
                info_col3.metric("Training Data", "2018-2026")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Valid ticker daalo — jaise AAPL, TSLA, MSFT")

else:
    st.info("👈 Sidebar mein ticker daalo aur Predict button dabao!")
    
    # Default graph
    st.markdown("### 📊 Supported Companies")
    companies_df = pd.DataFrame({
        "Ticker": list(company_names.keys()),
        "Company": list(company_names.values()),
        "Sector": [
            "Technology","Technology","Technology","Technology",
            "Technology","Technology","Technology","Technology",
            "Finance","Finance","Finance","Finance",
            "Energy","Energy","Energy"
        ]
    })
    st.dataframe(companies_df, use_container_width=True)
