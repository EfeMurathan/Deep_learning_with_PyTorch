#Burda MultiVariable seklinde bakicaz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

ticker = "MSFT" 
df = yf.download(ticker, start="2020-01-01")

features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
target = df[['Close']].values

seq_length = 30
split_idx = int(0.8 * (len(features) - seq_length)) + seq_length

train_features = features[:split_idx]
test_features = features[split_idx:]

train_target = target[:split_idx]
test_target = target[split_idx:]
feature_scaler = MinMaxScaler()
train_features_scaled = feature_scaler.fit_transform(train_features)
test_features_scaled = feature_scaler.transform(test_features)
scaled_features = np.vstack((train_features_scaled, test_features_scaled))

# Sadece Çıktı (Close) için Scaler
target_scaler = MinMaxScaler()
train_target_scaled = target_scaler.fit_transform(train_target)
test_target_scaled = target_scaler.transform(test_target)
scaled_target = np.vstack((train_target_scaled, test_target_scaled))

# 4. Zaman Serisi Dizilerini Oluşturma
X, y = [], []
for i in range(len(scaled_features) - seq_length):
    # X, son 30 günün 5 özelliğini (OHLCV) içerecek -> Boyut: (30, 5)
    X.append(scaled_features[i:i+seq_length])
    # y, 31. günün SADECE kapanış fiyatını içerecek -> Boyut: (1)
    y.append(scaled_target[i+seq_length])

X = np.array(X)
y = np.array(y)

# 5. Tensor'lara Çevirme (Train/Test Split)
train_size = int(0.8 * len(X))

X_train = torch.tensor(X[:train_size], dtype=torch.float32).to(device)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).to(device)

X_test = torch.tensor(X[train_size:], dtype=torch.float32).to(device)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).to(device)

print(f"Eğitim Girdisi (X_train) Şekli: {X_train.shape}") # Örn: (1311, 30, 5)

# 6. Çok Değişkenli LSTM Modeli
class PredictionModel(nn.Module):
    # input_dim 1'den 5'e çıkarıldı. 
    # Daha fazla bilgi girdiği için hidden_dim 32'den 64'e yükseltildi.
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
        # Sadece son zaman adımının çıktısını (yarının fiyatını) alıyoruz
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel().to(device)

# 7. Model Eğitimi
criterion = nn.MSELoss()
# Yeni mimari için öğrenme oranını (learning rate) biraz daha dengeli olan 0.005'e düşürdük
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

# 8. Test ve Değerlendirme (Evaluation)
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    y_train_pred = model(X_train)

# Sadece hedef değişkeni için kullandığımız target_scaler ile Ters Çevirme (Inverse Transform) yapıyoruz
y_train_pred_inv = target_scaler.inverse_transform(y_train_pred.cpu().numpy())
y_train_inv = target_scaler.inverse_transform(y_train.cpu().numpy())

y_test_pred_inv = target_scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_inv = target_scaler.inverse_transform(y_test.cpu().numpy())

# Hata Hesaplama
train_rmse = root_mean_squared_error(y_train_inv, y_train_pred_inv)
test_rmse = root_mean_squared_error(y_test_inv, y_test_pred_inv)

print(f"\nEğitim Seti (Train) RMSE: {train_rmse:.2f} $")
print(f"Test Seti (Test) RMSE:  {test_rmse:.2f} $")

# 9. Görselleştirme
fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(4, 1)

# Gerçek Fiyatlar ve Tahminler Grafiği
ax1 = fig.add_subplot(gs[:3, 0])
test_dates = df.index[-len(y_test_inv):]

ax1.plot(test_dates, y_test_inv, color="blue", label="Gerçek Kapanış (Actual Close)")
ax1.plot(test_dates, y_test_pred_inv, color="green", label="Tahmin Edilen (Predicted Close)")
ax1.legend()
ax1.set_title(f"{ticker} Hisse Senedi Fiyat Tahmini (OHLCV Modeli)", fontsize=16)
ax1.set_xlabel("Tarih")
ax1.set_ylabel("Fiyat ($)")
ax1.grid(True, alpha=0.3)

# Hata (Error) Grafiği
ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color="blue", linestyle="--", label="Test RMSE")
prediction_error = np.abs(y_test_inv.flatten() - y_test_pred_inv.flatten())
ax2.plot(test_dates, prediction_error, color="red", label="Tahmin Hatası (Error)")
ax2.legend()
ax2.set_title("Tahmin Hatası Miktarı ($)")
ax2.set_xlabel("Tarih")
ax2.set_ylabel("Hata ($)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()