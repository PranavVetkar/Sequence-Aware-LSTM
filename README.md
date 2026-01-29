# 🔁 Sequence-Aware LSTM for Crypto Price Prediction

A deep learning experiment that uses an **LSTM (Long Short-Term Memory) neural network** to model **temporal dependencies** in Bitcoin price data and predict the **next time-step price**.

This project explores whether **sequence-aware models** perform better than traditional regression approaches for financial time series.

---

## 🚀 What This Project Does

- Loads historical BTC price data
- Scales prices to a 0–1 range
- Creates rolling **time-window sequences**
- Trains a **stacked LSTM neural network**
- Predicts the **next-period BTC price**
- Demonstrates sequence learning in time-series data

---

## 🧠 Why LSTM?

Traditional ML models struggle with time dependency.  
**LSTMs are designed to learn patterns across sequences**, making them suitable for:
- Time-series data
- Market trends
- Temporal dependencies

This project tests that idea on crypto prices.

---

## 🧩 Model Architecture

- Input (10 time steps)
- LSTM (50 units, return_sequences=True)
- LSTM (50 units)
- Dense (25)
- Dense (1) → Predicted Price

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **TensorFlow / Keras**

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PranavVetkar/Sequence-Aware-LSTM.git
cd Sequence-Aware-LSTM
