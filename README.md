**📈 Stock Prediction Web Application**

A web-based application for stock analysis and prediction, built with Streamlit, Plotly, yfinance, and ARIMA modeling.
This app provides users with interactive stock charts, technical indicators, and up to 4 years of stock price forecasting!

**🚀 Features**

**📊 Stock Data Retrieval** — Fetches up to 10 years of historical stock data using the yfinance API.

**📈 Technical Indicators** — Calculates moving averages, daily change, trading volume, and 10-year high/low prices.

**🔮 ARIMA Prediction Model** — Forecasts stock prices up to 4 years into the future.

**🖥️ Interactive Visualizations** — Beautiful candlestick charts, forecast scatter plots, and volume bar charts using Plotly.

**⚙️ Customizable Settings** — Select stock ticker and prediction duration (1–4 years) via a user-friendly sidebar.



**🛠 Technologies Used**

**Python**

**Streamlit** — Web app framework

**yfinance **— Stock market data retrieval

**Plotly **— Interactive visualizations

**Statsmodels** — ARIMA model for time series forecasting

**Pandas and NumPy** — Data manipulation and processing

**🧠 Project Methodology**

**Data Retrieval:** Fetch historical stock data (open, high, low, close, volume) using yfinance.

**Data Preparation:** Format closing prices for ARIMA training.

**Model Training:** Train an ARIMA (5,1,0) model on closing prices.

**Prediction:** Forecast future stock prices monthly for the selected years.

**Visualization:**

Candlestick chart (historical)

Scatter plot (predicted)

Bar chart (trading volume)

Display Metrics:

Last Price

Daily Change

50-day and 200-day Moving Averages

10-Year High/Low

10-Year Average Volume

Current Trading Volume


**📋 How to Run Locally**
**1. Clone the repository:**

        git clone https://github.com/yourusername/stock-prediction-web-app.git
        cd stock-prediction-web-app

**2. Install dependencies:**

        pip install -r requirements.txt

**3. Run the application:**

        streamlit run main.py


**📄 License**

This project is for educational purposes.




