from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load LSTM model
model = load_model('model/model.h5')

# Load and scale data
df = pd.read_csv('btc.csv', index_col='Date', parse_dates=True)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

# Function to generate input-output pairs
def generate_input_output(df, window_size=15):
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(len(df_as_np) - window_size):
        row = df_as_np[i:i+window_size, 1:]  # Use columns from index 1 to the end (excluding 'Open')
        X.append(row)
        label = df_as_np[i+window_size, 0]  # Target 'Open'
        y.append(label)
    return np.array(X), np.array(y)

# Generate input-output pairs
WINDOW_SIZE = 15
X, y = generate_input_output(df_scaled, WINDOW_SIZE)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        num_future_steps = data['num_future_steps']

        # Generate predictions
        recent_data = X[-1].reshape(1, WINDOW_SIZE, X.shape[-1])
        predictions = []
        for _ in range(num_future_steps):
            prediction = model.predict(recent_data)[0][0]
            predictions.append(prediction)
            recent_data = np.append(recent_data[:, 1:, :], [[[0] * (X.shape[-1] - 1) + [prediction]]], axis=1)

        # Inverse transform the predictions
        dummy_df = np.zeros((num_future_steps, df_scaled.shape[1]))
        dummy_df[:, 0] = predictions
        inverse_predictions = scaler.inverse_transform(dummy_df)[:, 0].tolist()

        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=num_future_steps).tolist()

        # Plot predictions
        fig, ax = plt.subplots()
        ax.plot(future_dates, inverse_predictions, label='Predictions')
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return jsonify({'predictions': inverse_predictions, 'graph': image_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
