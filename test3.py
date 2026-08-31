import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# 1. Cihaz Ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 2. Veri İndirme 
ticker = "MSFT" 
df = yf.download(ticker, start="2020-01-01")

# 3. Yüzdelik Getiri (Returns) Hesaplaması *** EN KRİTİK DEĞİŞİKLİK ***
# Tüm verileri bir önceki güne göre yüzdelik değişime çeviriyoruz.
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[f'Ret_{col}'] = df[col].pct_change()

# İlk günün öncesi olmadığı için NaN (boş) değer oluşur, onu siliyoruz
df = df.dropna()

# Girdi Özelliklerimiz (Features) artık yüzdelik değişimler
features = df[['Ret_Open', 'Ret_High', 'Ret_Low', 'Ret_Close', 'Ret_Volume']].values
# Hedefimiz (Target) sadece Kapanışın yüzdelik değişimi
target = df[['Ret_Close']].values
# Son aşamada yüzdeleri fiyata çevirmek için ham kapanış fiyatlarını saklıyoruz
close_prices = df['Close'].values 

# 4. Veri Sızıntısını Önleyerek Ölçeklendirme (Train/Test Split & Scaling)
seq_length = 30
split_idx = int(0.8 * (len(features) - seq_length)) + seq_length

train_features = features[:split_idx]
test_features = features[split_idx:]
train_target = target[:split_idx]
test_target = target[split_idx:]

# Yüzdelik veriler için StandardScaler daha uygundur
feature_scaler = StandardScaler()
train_features_scaled = feature_scaler.fit_transform(train_features)
test_features_scaled = feature_scaler.transform(test_features)
scaled_features = np.vstack((train_features_scaled, test_features_scaled))

target_scaler = StandardScaler()
train_target_scaled = target_scaler.fit_transform(train_target)
test_target_scaled = target_scaler.transform(test_target)
scaled_target = np.vstack((train_target_scaled, test_target_scaled))

# 5. Zaman Serisi Dizilerini Oluşturma
X, y, prev_close = [], [], []
for i in range(len(scaled_features) - seq_length):
    X.append(scaled_features[i:i+seq_length])
    y.append(scaled_target[i+seq_length])
    # Tahmin edilen yüzdeyi fiyata çevirmek için, hedef günden BİR ÖNCEKİ GÜNÜN gerçek fiyatını kaydediyoruz
    prev_close.append(close_prices[i+seq_length-1])

X = np.array(X)
y = np.array(y)
prev_close = np.array(prev_close).reshape(-1, 1)

# 6. Tensor'lara Çevirme (Train/Test Split)
train_size = int(0.8 * len(X))

X_train = torch.tensor(X[:train_size], dtype=torch.float32).to(device)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).to(device)

X_test = torch.tensor(X[train_size:], dtype=torch.float32).to(device)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).to(device)

test_prev_close = prev_close[train_size:] # Test seti için önceki gün fiyatları

# 7. Model Tanımlaması (LSTM)
class PredictionModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel().to(device)

# 8. Model Eğitimi
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 200

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_train_pred = model(X_train)
    loss = criterion(y_train_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f'Epoch [{epoch}/{num_epochs-1}], Kayıp (Loss): {loss.item():.6f}')

# 9. Test ve Tahminleri Fiyata Geri Çevirme (Reconstruction)
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

# Önce StandardScaler'ı tersine çevirip YÜZDELİK TAHMİNLERİ buluyoruz
y_test_pred_return = target_scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_actual_return = target_scaler.inverse_transform(y_test.cpu().numpy())

# Yüzdelik tahminleri, Dolar cinsinden FİYATA çeviriyoruz: Fiyat_Dün * (1 + Getiri_Bugün)
predicted_prices = test_prev_close * (1 + y_test_pred_return)
actual_prices = test_prev_close * (1 + y_test_actual_return)

# Hata Hesaplama (Fiyat üzerinden)
test_rmse = root_mean_squared_error(actual_prices, predicted_prices)
print(f"\nTest Seti (Test) RMSE:  {test_rmse:.2f} $")

# 10. Görselleştirme
fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(4, 1)

ax1 = fig.add_subplot(gs[:3, 0])
test_dates = df.index[-len(actual_prices):]

ax1.plot(test_dates, actual_prices, color="blue", label="Gerçek Kapanış")
# Tahmin edilen fiyatları çizdiriyoruz
ax1.plot(test_dates, predicted_prices, color="green", label="Modelin Tahmini")
ax1.legend()
ax1.set_title(f"{ticker} Hisse Senedi Fiyat Tahmini (Yüzdelik Getiri Modeli)", fontsize=16)
ax1.set_xlabel("Tarih")
ax1.set_ylabel("Fiyat ($)")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color="blue", linestyle="--", label="Test RMSE")
prediction_error = np.abs(actual_prices.flatten() - predicted_prices.flatten())
ax2.plot(test_dates, prediction_error, color="red", label="Tahmin Hatası ($)")
ax2.legend()
ax2.set_title("Tahmin Hatası Miktarı ($)")
ax2.set_xlabel("Tarih")
ax2.set_ylabel("Hata ($)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()