# test5.py
# test4.py'nin üzerine inşa edildi.
# EK: Baseline Karşılaştırması
#
# Bir modelin gerçekten "öğrenip öğrenmediğini" anlamak için
# en basit tahmincilerle (baseline) karşılaştırılması şarttır.
# RMSE düşük görünse bile baseline'dan kötüyse model bir işe yaramamıştır.
#
# Kullanılan Baseline'lar:
#   1. Naive (Persistence): Yarın = Bugün  ← en saf tahmin
#   2. Hareketli Ortalama : Yarın = Son 5 günün ortalaması

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

# ─────────────────────────────────────────────
# 1. Cihaz Ayarı
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# ─────────────────────────────────────────────
# 2. Veri İndirme
# ─────────────────────────────────────────────
ticker = "MSFT"
df = yf.download(ticker, start="2020-01-01")

features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
target   = df[['Close']].values

# Ham close fiyatları — baseline hesabı için lazım (ölçeklenmemiş)
close_prices = df['Close'].values.flatten()

# ─────────────────────────────────────────────
# 3. Ölçeklendirme (sadece train'e fit — data leakage yok)
# ─────────────────────────────────────────────
seq_length = 30
n = len(features)

train_end = int(0.70 * n)
val_end   = int(0.80 * n)

feature_scaler = MinMaxScaler()
feature_scaler.fit(features[:train_end])

target_scaler = MinMaxScaler()
target_scaler.fit(target[:train_end])

features_scaled = feature_scaler.transform(features)
target_scaled   = target_scaler.transform(target)

# ─────────────────────────────────────────────
# 4. Sequence Oluştur → Böl (veri kaybı yok)
# ─────────────────────────────────────────────
def make_sequences(feat, tgt, seq_len):
    X, y = [], []
    for i in range(len(feat) - seq_len):
        X.append(feat[i : i + seq_len])
        y.append(tgt[i + seq_len])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(features_scaled, target_scaled, seq_length)

train_seq_end = train_end - seq_length
val_seq_end   = val_end   - seq_length

X_train_np, y_train_np = X_all[:train_seq_end],            y_all[:train_seq_end]
X_val_np,   y_val_np   = X_all[train_seq_end:val_seq_end], y_all[train_seq_end:val_seq_end]
X_test_np,  y_test_np  = X_all[val_seq_end:],              y_all[val_seq_end:]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_val   = torch.tensor(X_val_np,   dtype=torch.float32)
y_val   = torch.tensor(y_val_np,   dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32).to(device)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. Baseline Hesabı  ★ YENİ ★
# ─────────────────────────────────────────────
# Baseline için ölçeklenmemiş ham fiyatları kullanıyoruz.
# Her sequence i için hedef: close_prices[seq_length + i]
# Yani test seti hedefleri: close_prices[seq_length + val_seq_end :]

# Test seti gerçek fiyatları (ham, ölçeklenmemiş)
test_actual_prices = close_prices[seq_length + val_seq_end:]

# -- Baseline 1: Naive (Persistence) --
# "Yarın bugünkü fiyatla aynı olacak"
# Her sequence'ın son günündeki close = X_test'in son zaman adımının Close sütunu
# Close sütunu OHLCV'de index 3
last_close_scaled = X_test_np[:, -1, 3].reshape(-1, 1)   # ölçeklenmiş

# Feature scaler 5 özellik için fit edildi, sadece Close'u inverse etmek için
# dummy 5-sütunlu matris oluştur, Close yerine değeri koy, geri kalanı 0
dummy = np.zeros((len(last_close_scaled), 5))
dummy[:, 3] = last_close_scaled.flatten()
naive_pred_prices = feature_scaler.inverse_transform(dummy)[:, 3]

# -- Baseline 2: 5 Günlük Hareketli Ortalama --
# "Yarın son 5 günün ortalaması kadar olacak"
# Her sequence'ın son 5 gününün Close ortalaması
last_5_close_scaled = X_test_np[:, -5:, 3]               # (N, 5) ölçeklenmiş
ma5_scaled = last_5_close_scaled.mean(axis=1).reshape(-1, 1)

dummy_ma = np.zeros((len(ma5_scaled), 5))
dummy_ma[:, 3] = ma5_scaled.flatten()
ma5_pred_prices = feature_scaler.inverse_transform(dummy_ma)[:, 3]

# Baseline RMSE
naive_rmse = root_mean_squared_error(test_actual_prices, naive_pred_prices)
ma5_rmse   = root_mean_squared_error(test_actual_prices, ma5_pred_prices)

# ─────────────────────────────────────────────
# 6. DataLoader
# ─────────────────────────────────────────────
BATCH_SIZE   = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────
# 7. Model
# ─────────────────────────────────────────────
class PredictionModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

model = PredictionModel().to(device)

# ─────────────────────────────────────────────
# 8. Optimizer & Scheduler
# ─────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ─────────────────────────────────────────────
# 9. Eğitim
# ─────────────────────────────────────────────
num_epochs    = 100
train_losses  = []
val_losses    = []
best_val_loss = float('inf')
MODEL_PATH    = "stock_model_v2.pth"

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())

    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)

    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_batch_losses.append(criterion(model(X_batch), y_batch).item())

    avg_val_loss = np.mean(val_batch_losses)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

    scheduler.step(avg_val_loss)

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        marker = " ← best" if avg_val_loss == best_val_loss else ""
        print(f"Epoch [{epoch:>3}/{num_epochs-1}] | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")

# ─────────────────────────────────────────────
# 10. En İyi Modeli Yükle
# ─────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH))
print(f"\nEn iyi model yüklendi (best val loss: {best_val_loss:.6f})")

# ─────────────────────────────────────────────
# 11. Model Tahminleri
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_test_pred  = model(X_test)
    y_train_pred = model(X_train.to(device))

y_test_pred_inv = target_scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_inv      = target_scaler.inverse_transform(y_test.cpu().numpy())

y_train_pred_inv = target_scaler.inverse_transform(y_train_pred.cpu().numpy())
y_train_inv      = target_scaler.inverse_transform(y_train.numpy())

train_rmse = root_mean_squared_error(y_train_inv, y_train_pred_inv)
test_rmse  = root_mean_squared_error(y_test_inv,  y_test_pred_inv)

# ─────────────────────────────────────────────
# 12. Baseline Karşılaştırma Raporu  ★ YENİ ★
# ─────────────────────────────────────────────
print(f"\n{'═'*52}")
print(f"  {'MODEL / BASELINE':<28} {'Test RMSE ($)':>12}  {'Sonuç'}")
print(f"{'─'*52}")
print(f"  {'Naive (Yarın = Bugün)':<28} {naive_rmse:>12.2f}  (referans)")
print(f"  {'Hareketli Ort. (MA-5)':<28} {ma5_rmse:>12.2f}  "
      f"{'✅ baseline\'den iyi' if ma5_rmse < naive_rmse else '❌ naive\'den kötü'}")
print(f"  {'LSTM Modelimiz':<28} {test_rmse:>12.2f}  "
      f"{'✅ baseline\'den iyi' if test_rmse < naive_rmse else '❌ baseline\'den kötü!'}")
print(f"{'─'*52}")

if test_rmse < naive_rmse:
    kazanim = ((naive_rmse - test_rmse) / naive_rmse) * 100
    print(f"  Model, naive baseline'a göre %{kazanim:.1f} daha iyi tahmin yapıyor.")
else:
    kayip = ((test_rmse - naive_rmse) / naive_rmse) * 100
    print(f"  ⚠️  Model, naive baseline'dan %{kayip:.1f} DAHA KÖTÜ!")
    print(f"  ⚠️  Modeli yeniden değerlendirmek gerekebilir.")
print(f"{'═'*52}\n")

# ─────────────────────────────────────────────
# 13. Gerçek vs Tahmin Tablosu (son 10 gün)
# ─────────────────────────────────────────────
test_dates = df.index[seq_length + val_seq_end:]

results_df = pd.DataFrame({
    "Gerçek $"      : test_actual_prices.round(2),
    "LSTM $"        : y_test_pred_inv.flatten().round(2),
    "Naive $"       : naive_pred_prices.round(2),
    "MA-5 $"        : ma5_pred_prices.round(2),
    "LSTM Hata $"   : np.abs(test_actual_prices - y_test_pred_inv.flatten()).round(2),
    "Naive Hata $"  : np.abs(test_actual_prices - naive_pred_prices).round(2),
}, index=test_dates)

print("── Son 10 Gün: Tüm Modeller ──")
print(results_df.tail(10).to_string())

# ─────────────────────────────────────────────
# 14. Yarınki Tahmin
# ─────────────────────────────────────────────
last_30 = feature_scaler.transform(
    df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-seq_length:]
)
last_30_tensor = torch.tensor(last_30, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    next_scaled = model(last_30_tensor)

next_day_price  = float(target_scaler.inverse_transform(next_scaled.cpu().numpy())[0, 0])
last_real_price = float(df['Close'].values[-1])

print(f"\n{'─'*40}")
print(f"  Son Gerçek Fiyat          : {last_real_price:.2f} $  ({df.index[-1].date()})")
print(f"  LSTM Tahmini (yarın)      : {next_day_price:.2f} $")
change = next_day_price - last_real_price
arrow  = "▲" if change >= 0 else "▼"
print(f"  Beklenen Değişim          : {arrow} {abs(change):.2f} $ ({change/last_real_price*100:.2f}%)")
print(f"  Naive Tahmini (yarın)     : {last_real_price:.2f} $ (değişmez)")
print(f"{'─'*40}\n")

# ─────────────────────────────────────────────
# 15. Görselleştirme
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 14))
gs  = fig.add_gridspec(5, 1)

# — Fiyat Karşılaştırması —
ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(test_dates, test_actual_prices,         color="blue",   lw=1.5, label="Gerçek Fiyat")
ax1.plot(test_dates, y_test_pred_inv.flatten(),  color="green",  lw=1.5, label=f"LSTM  (RMSE: {test_rmse:.2f}$)")
ax1.plot(test_dates, naive_pred_prices,          color="orange", lw=1.0, linestyle="--", label=f"Naive (RMSE: {naive_rmse:.2f}$)")
ax1.plot(test_dates, ma5_pred_prices,            color="purple", lw=1.0, linestyle=":",  label=f"MA-5  (RMSE: {ma5_rmse:.2f}$)")
ax1.legend()
ax1.set_title(f"{ticker} — LSTM vs Baseline Karşılaştırması", fontsize=14)
ax1.set_xlabel("Tarih")
ax1.set_ylabel("Fiyat ($)")
ax1.grid(True, alpha=0.3)

# — Hata Karşılaştırması —
ax2 = fig.add_subplot(gs[3, 0])
lstm_error  = np.abs(test_actual_prices - y_test_pred_inv.flatten())
naive_error = np.abs(test_actual_prices - naive_pred_prices)
ax2.plot(test_dates, lstm_error,  color="green",  lw=1.2, label=f"LSTM Hatası  (ort: {lstm_error.mean():.2f}$)")
ax2.plot(test_dates, naive_error, color="orange", lw=1.0, linestyle="--", label=f"Naive Hatası (ort: {naive_error.mean():.2f}$)")
ax2.axhline(test_rmse,  color="green",  linestyle=":", alpha=0.6)
ax2.axhline(naive_rmse, color="orange", linestyle=":", alpha=0.6)
ax2.legend()
ax2.set_title("Mutlak Tahmin Hatası Karşılaştırması ($)")
ax2.set_xlabel("Tarih")
ax2.set_ylabel("Hata ($)")
ax2.grid(True, alpha=0.3)

# — Loss Eğrisi —
ax3 = fig.add_subplot(gs[4, 0])
ax3.plot(train_losses, color="blue",   label="Train Loss")
ax3.plot(val_losses,   color="orange", label="Validation Loss")
ax3.legend()
ax3.set_title("Eğitim & Validation Kayıp Eğrisi")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("MSE Loss")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
