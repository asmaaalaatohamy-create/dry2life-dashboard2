import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

print("=" * 60)
print("🚀 DRY2LIFE - LSTM Model Training Script")
print("=" * 60)

# ============================================
# 1. توليد بيانات محاكاة (لأنه مش عندك ملف CSV)
# ============================================
print("\n📊 Generating synthetic time-series data...")

np.random.seed(42)

days = 365
time_steps = np.arange(days)

base_salinity = 8 + 4 * np.sin(time_steps / 30) + 2 * np.sin(time_steps / 90)
noise_salinity = np.random.normal(0, 0.5, days)
salinity = np.maximum(0.5, base_salinity + noise_salinity)

base_moisture = 50 + 15 * np.sin(time_steps / 40 + 1.2) + 10 * np.sin(time_steps / 100)
noise_moisture = np.random.normal(0, 3, days)
moisture = np.clip(base_moisture + noise_moisture, 20, 80)

base_temp = 28 + 10 * np.sin(time_steps / 50 + 0.5)
noise_temp = np.random.normal(0, 1.5, days)
temperature = np.clip(base_temp + noise_temp, 15, 45)

df = pd.DataFrame({
    'salinity': salinity,
    'moisture': moisture,
    'temperature': temperature
})

print(f"✅ Generated {len(df)} rows of synthetic data.")
print(df.head())

# ============================================
# 2. تطبيع البيانات (Normalization)
# ============================================
print("\n🔄 Normalizing data...")

features = ['salinity', 'moisture', 'temperature']
data = df[features].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print(f"✅ Data normalized. Shape: {scaled_data.shape}")

# ============================================
# 3. إنشاء النوافذ الزمنية (Time Windows)
# ============================================
print("\n🔧 Creating time windows (sequence length = 7 days)...")

def create_sequences(data, n_steps=7):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

n_steps = 7
X, y = create_sequences(scaled_data, n_steps)

print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

# ============================================
# 4. تقسيم البيانات (تدريب / اختبار)
# ============================================
print("\n📂 Splitting data into train/test sets (80/20)...")

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# ============================================
# 5. بناء نموذج LSTM
# ============================================
print("\n🏗️ Building LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(n_steps, X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
print(model.summary())

# ============================================
# 6. تدريب النموذج
# ============================================
print("\n🚀 Training LSTM model...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ============================================
# 7. تقييم النموذج
# ============================================
print("\n📊 Evaluating model on test set...")

loss, mae = model.evaluate(X_test, y_test, verbose=0)

y_pred_scaled = model.predict(X_test, verbose=0)
y_test_actual = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), 
                    np.zeros((len(y_test), 2))], axis=1)
)[:, 0]
y_pred_actual = scaler.inverse_transform(
    np.concatenate([y_pred_scaled, 
                    np.zeros((len(y_pred_scaled), 2))], axis=1)
)[:, 0]

from sklearn.metrics import r2_score
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"✅ Test Loss (MSE): {loss:.4f}")
print(f"✅ Test MAE: {mae:.4f}")
print(f"✅ Test R² Score: {r2:.4f}")

# ============================================
# 8. حفظ النموذج والـ Scaler
# ============================================
print("\n💾 Saving model and scaler...")

model.save("lstm_salinity_model.h5")
print("✅ Model saved as 'lstm_salinity_model.h5'")

joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as 'scaler.pkl'")

# ============================================
# 9. اختبار سريع للنموذج المحفوظ
# ============================================
print("\n🧪 Testing loaded model with a sample prediction...")

from tensorflow.keras.models import load_model

loaded_model = load_model("lstm_salinity_model.h5")
loaded_scaler = joblib.load("scaler.pkl")

last_7 = scaled_data[-7:].reshape(1, 7, 3)
pred_scaled = loaded_model.predict(last_7, verbose=0)

dummy_pred = np.zeros((1, 3))
dummy_pred[0, 0] = pred_scaled[0, 0]
pred_actual = loaded_scaler.inverse_transform(dummy_pred)[0, 0]

print(f"🔮 Sample Prediction for next day's salinity: {pred_actual:.2f} dS/m")
print(f"📌 Last actual salinity: {df['salinity'].iloc[-1]:.2f} dS/m")

print("\n" + "=" * 60)
print("✅ Training complete! Files are ready.")
print("📁 'lstm_salinity_model.h5' and 'scaler.pkl' created successfully.")
print("=" * 60)