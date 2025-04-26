import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period='10y'):
    """Fetch historical stock data using yfinance with 10-year history"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df, stock.info

def prepare_data(df):
    """Prepare data for ARIMA model"""
    return pd.DataFrame(df['Close']).reset_index()

def train_arima_model(data, order=(5,1,0)):
    """Train ARIMA model on the closing prices"""
    model = ARIMA(data['Close'], order=order)
    return model.fit()

def make_predictions(model, years):
    """Generate future predictions using the trained model"""
    months = years * 12
    forecast = model.forecast(steps=months)
    return forecast

def plot_stock_data(historical_data, forecast_data, ticker, prediction_years):
    """Create interactive plots using Plotly with optimizations for longer time periods"""
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        vertical_spacing=0.2,
                        row_heights=[0.7, 0.3])

    # Calculate yearly candlesticks for better visualization of long-term data
    yearly_data = historical_data.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Stock price plot with forecast
    fig.add_trace(
        go.Candlestick(
            x=yearly_data.index,
            open=yearly_data['Open'],
            high=yearly_data['High'],
            low=yearly_data['Low'],
            close=yearly_data['Close'],
            name='Historical'
        ),
        row=1, col=1
    )

    # Add forecast as dots
    forecast_dates = pd.date_range(
        start=historical_data.index[-1] + timedelta(days=1), 
        periods=len(forecast_data),
        freq='M'
    )
    
    # Resample forecast data to monthly
    forecast_monthly = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_data
    }).set_index('Date')
    
    fig.add_trace(
        go.Scatter(
            x=forecast_monthly.index,
            y=forecast_monthly['Forecast'],
            mode='markers',
            name=f'{prediction_years}-Year Forecast',
            marker=dict(
                color='white',
                size=8,
                symbol='circle',
                line=dict(
                    color='black',
                    width=1
                )
            )
        ),
        row=1, col=1
    )

    # Volume bar chart with yearly aggregation
    fig.add_trace(
        go.Bar(
            x=yearly_data.index,
            y=yearly_data['Volume'],
            name='Volume'
        ),
        row=2, col=1
    )

    # Update layout with dark theme and better date handling
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"{ticker} Stock Analysis (10-Year History) with {prediction_years}-Year Prediction",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Update axes with better date formatting and grid
    fig.update_xaxes(
        gridcolor='rgba(128,128,128,0.2)',
        zerolinecolor='rgba(128,128,128,0.2)',
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="10Y")
            ])
        )
    )
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.2)')

    return fig

def calculate_metrics(historical_data):
    """Calculate basic technical indicators with additional long-term metrics"""
    return {
        'Last Price': historical_data['Close'][-1],
        'Daily Change': f"{((historical_data['Close'][-1] / historical_data['Close'][-2]) - 1) * 100:.2f}%",
        '50-day MA': historical_data['Close'].rolling(window=50).mean()[-1],
        '200-day MA': historical_data['Close'].rolling(window=200).mean()[-1],
        'Trading Volume': historical_data['Volume'][-1],
        '10Y High': historical_data['High'].max(),
        '10Y Low': historical_data['Low'].min(),
        '10Y Avg Volume': historical_data['Volume'].mean()
    }

def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")
    
    st.title("📈 Stock Prediction Web Application")

    # Sidebar
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
    st.sidebar.markdown("### Forecast Settings")
    prediction_years = st.sidebar.slider(
        "Prediction Years",
        min_value=1,
        max_value=4,
        value=1,
        help="Select the number of years to forecast (max 4 years)"
    )
    
    try:
        # Fetch Data
        with st.spinner("Fetching 10 years of stock data..."):
            df, stock_info = fetch_stock_data(ticker)
            
        # Display company info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Company Info")
            st.write(f"**{stock_info.get('longName', ticker)}**")
            st.write(f"Sector: {stock_info.get('sector', 'N/A')}")
            st.write(f"Industry: {stock_info.get('industry', 'N/A')}")
        
        # Calculate and display metrics
        metrics = calculate_metrics(df)
        with col2:
            st.subheader("Key Metrics")
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:,.2f}" if isinstance(value, float) else f"{metric}: {value}")
        
        # Train ARIMA model
        with st.spinner("Training model and generating predictions..."):
            data = prepare_data(df)
            model = train_arima_model(data)
            forecast = make_predictions(model, years=prediction_years)
        
        # Plot results
        st.subheader(f"Stock Analysis and {prediction_years}-Year Prediction")
        fig = plot_stock_data(df, forecast, ticker, prediction_years)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast data
        st.subheader(f"Past 1-Year Stock Data (Monthly)")
        past_1y_data = df.last('1Y').resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        st.dataframe(past_1y_data)

        st.subheader(f"{prediction_years}-Year Price Forecasts (Monthly)")
        forecast_dates = pd.date_range(
            start=df.index[-1] + timedelta(days=1),
            periods=len(forecast),
            freq='M'
        )
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Price': forecast
        })
        forecast_df.set_index('Date', inplace=True)
        st.dataframe(forecast_df)
        
    except Exception as e:
        st.markdown(
        """
        <div style="
            color: red; 
            text-align: center; 
            font-size: 24px;
        ">
            Please check the stock ticker and try again.
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()