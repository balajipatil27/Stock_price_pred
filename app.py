import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Remove TensorFlow import since it's causing issues
# from tensorflow.keras.models import load_model

from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

def simple_prediction(data, window=20):
    """
    Simple prediction using moving average
    This replaces the deep learning model temporarily
    """
    predictions = []
    for i in range(window, len(data)):
        # Use moving average of last 'window' days for prediction
        pred = np.mean(data[i-window:i])
        predictions.append(pred)
    
    return np.array(predictions)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'AAPL'  # Changed default to a more common stock
        
        try:
            # Define the start and end dates for stock data
            start = dt.datetime(2020, 1, 1)  # Reduced date range for faster loading
            end = dt.datetime(2024, 10, 1)
            
            # Download stock data
            print(f"Downloading data for {stock}...")
            df = yf.download(stock, start=start, end=end)
            
            if df.empty:
                return render_template('index.html', error=f"No data found for stock symbol: {stock}")
            
            print(f"Data downloaded successfully. Shape: {df.shape}")
            print(f"Columns available: {df.columns.tolist()}")
            
            # Handle multi-level columns if they exist
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1)
            
            # Ensure we have the required columns
            if 'Close' not in df.columns:
                return render_template('index.html', error=f"Close price data not available for {stock}")
            
            # Descriptive Data
            data_desc = df.describe()
            
            # Exponential Moving Averages
            ema20 = df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = df['Close'].ewm(span=50, adjust=False).mean()
            ema100 = df['Close'].ewm(span=100, adjust=False).mean()
            ema200 = df['Close'].ewm(span=200, adjust=False).mean()
            
            # Data splitting
            split_point = int(len(df) * 0.70)
            data_training = pd.DataFrame(df['Close'][:split_point])
            data_testing = pd.DataFrame(df['Close'][split_point:])
            
            print(f"Training data shape: {data_training.shape}")
            print(f"Testing data shape: {data_testing.shape}")
            
            # Simple prediction using moving average (replacing ML model)
            print("Generating predictions...")
            test_data = data_testing['Close'].values
            full_data = df['Close'].values
            
            # Generate predictions for the test period
            predictions = []
            train_size = len(data_training)
            
            for i in range(train_size, len(full_data)):
                if i >= 20:  # Need at least 20 days for moving average
                    pred = np.mean(full_data[i-20:i])
                    predictions.append(pred)
                else:
                    predictions.append(full_data[i-1] if i > 0 else full_data[0])
            
            y_predicted = np.array(predictions)
            y_test = test_data
            
            # Make sure arrays have same length
            min_len = min(len(y_predicted), len(y_test))
            y_predicted = y_predicted[:min_len]
            y_test = y_test[:min_len]
            
            print(f"Final prediction shape: {y_predicted.shape}")
            print(f"Final test data shape: {y_test.shape}")
            
            if len(y_predicted) == 0 or len(y_test) == 0:
                return render_template('index.html', error="Not enough data for meaningful predictions")
            
            # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], 'b', label='Closing Price', alpha=0.7)
            plt.plot(df.index, ema20, 'g', label='EMA 20', alpha=0.8)
            plt.plot(df.index, ema50, 'r', label='EMA 50', alpha=0.8)
            plt.title(f"{stock} - Closing Price vs Time (20 & 50 Days EMA)")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_chart_path = "static/ema_20_50.png"
            plt.savefig(ema_chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], 'b', label='Closing Price', alpha=0.7)
            plt.plot(df.index, ema100, 'g', label='EMA 100', alpha=0.8)
            plt.plot(df.index, ema200, 'r', label='EMA 200', alpha=0.8)
            plt.title(f"{stock} - Closing Price vs Time (100 & 200 Days EMA)")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_chart_path_100_200 = "static/ema_100_200.png"
            plt.savefig(ema_chart_path_100_200, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Plot 3: Prediction vs Original Trend
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(y_test)), y_test, 'g', label="Actual Price", linewidth=2, alpha=0.8)
            plt.plot(range(len(y_predicted)), y_predicted, 'r', label="Predicted Price (Moving Average)", linewidth=2, alpha=0.8)
            plt.title(f"{stock} - Simple Prediction vs Actual Trend")
            plt.xlabel("Days (Test Period)")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            prediction_chart_path = "stock_prediction.png"
            plt.savefig(prediction_chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Save dataset as CSV
            csv_file_path = f"static/{stock.replace('.', '_')}_dataset.csv"
            df.to_csv(csv_file_path)

            print("Charts generated successfully!")

            # Return the rendered template with charts and dataset
            return render_template('index.html', 
                                   stock_symbol=stock,
                                   plot_path_ema_20_50=ema_chart_path, 
                                   plot_path_ema_100_200=ema_chart_path_100_200, 
                                   plot_path_prediction=prediction_chart_path, 
                                   data_desc=data_desc.to_html(classes='table table-bordered table-striped'),
                                   dataset_link=csv_file_path,
                                   csv_filename=f"{stock.replace('.', '_')}_dataset.csv",
                                   prediction_method="Moving Average (Simple Prediction)")
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(f"static/{filename}", as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Make sure you have the following packages installed:")
    print("pip install flask pandas numpy matplotlib yfinance scikit-learn")
    app.run(debug=True, host='0.0.0.0', port=5000)

