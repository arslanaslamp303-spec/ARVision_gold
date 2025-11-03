import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- LSTM model class ---
class GoldLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=1):
        super(GoldLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Load data ---
df = pd.read_csv("xauusd_clean.csv", index_col=0, parse_dates=True)
features = ['Open','High','Low','Close','RSI','MACD','MACD_signal','MACD_diff']

# --- Normalize ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled, columns=features, index=df.index)

# --- Create sequences ---
def create_sequences(data, window=30):
    X = []
    for i in range(len(data) - window):
        X.append(data.iloc[i:i+window].values)
    return np.array(X)

X_all = create_sequences(scaled_df)
X_tensor = torch.tensor(X_all, dtype=torch.float32)

# --- Load model ---
model = GoldLSTM(input_size=8)
model.load_state_dict(torch.load("gold_lstm_model.pth", map_location="cpu"))
model.eval()

# --- Predict ---
with torch.no_grad():
    preds = model(X_tensor).squeeze().numpy()

# --- Denormalize predicted Close prices ---
close_min = df["Close"].min()
close_max = df["Close"].max()
preds_real = preds * (close_max - close_min) + close_min

# --- Streamlit UI ---
st.title("🏆 ARVision Gold – LSTM Forecast Dashboard")

st.write("### Historical Gold Prices (Close)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Actual"))
fig.add_trace(go.Scatter(x=df.index[-len(preds_real):], y=preds_real, name="Predicted", line=dict(color='orange')))
st.plotly_chart(fig, use_container_width=True)

st.write("### Next-day Forecast Example")
st.write(f"Predicted next close: **{preds_real[-1]:.2f} USD**")

