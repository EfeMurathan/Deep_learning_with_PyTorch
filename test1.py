import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

# 1. Cihaz Ayarı (GPU varsa GPU, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 2. Veri İndirme
ticker = "MSFT" 
df = yf.download(ticker, start="2020-01-01")

# Kapanış fiyatlarını numpy dizisine çevirelim
data_values = df['Close'].values.reshape(-1, 1)

# 3. Veri Sızıntısını (Data Leakage) Önleyerek Veriyi Ölçeklendirme
seq_length = 30

# Eğitim setinin bitiş indeksini belirliyoruz
split_idx = int(0.8 * (len(data_values) - seq_length)) + seq_length

train_data = data_values[:split_idx]
test_data = data_values[split_idx:]

scaler = MinMaxScaler()
# Scaler SADECE eğitim verisine fit edilmeli!
train_scaled = scaler.fit_transform(train_data)
# Test verisi eğitim verisinin min-max değerlerine göre transform ediliyor.
test_scaled = scaler.transform(test_data)

# Ölçeklenmiş veriyi tekrar birleştiriyoruz (Dizi oluşturmak için)
scaled_data = np.vstack((train_scaled, test_scaled))

# 4. Zaman Serisi Dizilerini (Sequences) Oluşturma
X, y = [], []
for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

X = np.array(X)
y = np.array(y)

# 5. Train ve Test Setlerini Ayırma
train_size = int(0.8 * len(X))

X_train = torch.tensor(X[:train_size], dtype=torch.float32).to(device)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).to(device)

X_test = torch.tensor(X[train_size:], dtype=torch.float32).to(device)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).to(device)

# 6. Model Tanımlaması
class PredictionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Dropout ekleyerek overfitting'i azaltıyoruz
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Sadece son zaman adımının çıktısını alıyoruz
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel().to(device)

# 7. Model Eğitimi (Training)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

# Modeli eğitim moduna alıyoruz
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    y_train_pred = model(X_train)
    loss = criterion(y_train_pred, y_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f'Epoch [{epoch}/{num_epochs-1}], Loss: {loss.item():.6f}')

# 8. Test ve Değerlendirme (Evaluation)
# Modeli değerlendirme moduna alıyoruz (Eğitim tamamlandı)
model.eval()
with torch.no_grad(): # Test sırasında Gradient hesaplamasını kapatarak bellek tasarrufu sağlıyoruz
    y_test_pred = model(X_test)
    y_train_pred = model(X_train)

# Değerleri orjinal fiyat skalasına geri döndürüyoruz
y_train_pred_inv = scaler.inverse_transform(y_train_pred.cpu().numpy())
y_train_inv = scaler.inverse_transform(y_train.cpu().numpy())

y_test_pred_inv = scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_inv = scaler.inverse_transform(y_test.cpu().numpy())

# RMSE Hesaplama
train_rmse = root_mean_squared_error(y_train_inv, y_train_pred_inv)
test_rmse = root_mean_squared_error(y_test_inv, y_test_pred_inv)

print(f"\nTrain RMSE: {train_rmse:.2f}")
print(f"Test RMSE:  {test_rmse:.2f}")

# 9. Görselleştirme
fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(4, 1)

# Gerçek Fiyatlar ve Tahminler Grafiği
ax1 = fig.add_subplot(gs[:3, 0])
# Test verisinin tarihlerini alalım
test_dates = df.index[-len(y_test_inv):]

ax1.plot(test_dates, y_test_inv, color="blue", label="Gerçek Fiyat (Actual)")
ax1.plot(test_dates, y_test_pred_inv, color="green", label="Tahmin Edilen Fiyat (Predicted)")
ax1.legend()
ax1.set_title(f"{ticker} Hisse Senedi Fiyat Tahmini", fontsize=16)
ax1.set_xlabel("Tarih")
ax1.set_ylabel("Fiyat ($)")
ax1.grid(True, alpha=0.3)

# Hata (Error) Grafiği
ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color="blue", linestyle="--", label="Test RMSE")
prediction_error = np.abs(y_test_inv.flatten() - y_test_pred_inv.flatten())
ax2.plot(test_dates, prediction_error, color="red", label="Mutlak Tahmin Hatası (Error)")
ax2.legend()
ax2.set_title("Tahmin Hatası Miktarı ($)")
ax2.set_xlabel("Tarih")
ax2.set_ylabel("Hata ($)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()