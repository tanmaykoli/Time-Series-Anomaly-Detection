# Anomaly Detection in Time Series Data (Autoencoder)

### 🚀 Week 4 Internship Project
This project implements an Unsupervised Learning approach to detect anomalies in time-series data using a **Deep Learning Autoencoder** built with TensorFlow/Keras.

## 📊 Dataset
The project uses the **Numenta Anomaly Benchmark (NAB)** dataset, specifically the `ambient_temperature_system_failure.csv` data. It contains temperature sensor readings where known system failures occurred.

## 🧠 Model Architecture
The model is a Sequential Autoencoder that compresses a 24-hour window of data into a lower-dimensional representation and then reconstructs it.
- **Input:** 24-step windowed time series.
- **Encoder:** Flatten layer followed by a Dense layer (16 units, ReLU).
- **Decoder:** Dense layer (24 units, Sigmoid) reshaped back to original dimensions.
- **Loss Function:** Mean Squared Error (MSE).

## 🛠️ How it Works
1. **Preprocessing:** Data is normalized using `MinMaxScaler` (0 to 1).
2. **Windowing:** The data is segmented into 24-hour sliding windows.
3. **Training:** The model trains on the first 60% of the data (assumed to be mostly normal).
4. **Thresholding:** Anomalies are detected by calculating the **Reconstruction Error**. If the error for a specific window exceeds the 99th percentile of training errors, it is flagged as an anomaly.

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn

## 📈 Results
The notebook generates a reconstruction error plot. Peaks above the dashed threshold line represent detected anomalies in the temperature system.

---

