# test4.py
# test2.py'nin üzerine inşa edildi.
# Eklenen PyTorch kavramları:
#   1. TensorDataset + DataLoader  (mini-batch eğitimi)
#   2. Train / Validation / Test ayrımı
#   3. Gradient Clipping             (LSTM'de exploding gradient önleme)
#   4. ReduceLROnPlateau Scheduler   (öğrenme oranını otomatik düşürme)
#   5. Model kaydetme & yükleme      (torch.save / torch.load)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader  # YENİ

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

# ─────────────────────────────────────────────
# 3. Ölçeklendirme (Data Leakage yok — sadece train'e fit)
# ─────────────────────────────────────────────
seq_length = 30
n = len(features)

train_end = int(0.70 * n)
val_end   = int(0.80 * n)

# Scaler'ı SADECE train kısmına fit et — leakage yok ✅
feature_scaler = MinMaxScaler()
feature_scaler.fit(features[:train_end])

target_scaler = MinMaxScaler()
target_scaler.fit(target[:train_end])

# Tüm veriyi dönüştür (fit sadece train'e yapıldı)
features_scaled = feature_scaler.transform(features)
target_scaled   = target_scaler.transform(target)

# ─────────────────────────────────────────────
# 4. Tüm Veriden Sequence Oluştur → Sonra Böl  (DÜZELTME)
# ─────────────────────────────────────────────
# ESKİ (yanlış): Her split'ten ayrı ayrı sequence üret
#   → Val ve test'in başındaki ilk 30 günün bağlamı kayboluyordu ❌
#
# YENİ (doğru): Önce TÜM veriden sequence üret, sonra böl
#   → Val'ın ilk sequence'ı train'in son 29 gününü görebilir ✅
def make_sequences(feat, tgt, seq_len):
    X, y = [], []
    for i in range(len(feat) - seq_len):
        X.append(feat[i : i + seq_len])
        y.append(tgt[i + seq_len])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(features_scaled, target_scaled, seq_length)

# Sequence indekslerini orijinal veri sınırlarına göre böl
# (seq_length kadar offset var çünkü her sequence seq_length günden oluşuyor)
train_seq_end = train_end - seq_length
val_seq_end   = val_end   - seq_length

X_train_np, y_train_np = X_all[:train_seq_end],              y_all[:train_seq_end]
X_val_np,   y_val_np   = X_all[train_seq_end:val_seq_end],   y_all[train_seq_end:val_seq_end]
X_test_np,  y_test_np  = X_all[val_seq_end:],                y_all[val_seq_end:]

# NumPy → PyTorch Tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_val   = torch.tensor(X_val_np,   dtype=torch.float32)
y_val   = torch.tensor(y_val_np,   dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32).to(device)

print(f"X_train shape: {X_train.shape} | X_val shape: {X_val.shape} | X_test shape: {X_test.shape}")

# ─────────────────────────────────────────────
# 6. DataLoader  (YENİ — en önemli PyTorch kavramlarından biri)
# ─────────────────────────────────────────────
# test2'de tüm eğitim verisi tek seferde (full-batch) modele veriliyordu.
# DataLoader bunu otomatik olarak küçük parçalara (batch) böler.
# shuffle=True: her epoch'ta veri karıştırılır → daha iyi genelleme.
BATCH_SIZE = 64

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val,   y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────
# 7. Model (test2 ile aynı mimari, değişiklik yok)
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
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel().to(device)

# ─────────────────────────────────────────────
# 8. Loss, Optimizer, Scheduler  (Scheduler YENİ)
# ─────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ReduceLROnPlateau: validation loss belirli epoch sayısı (patience) boyunca
# iyileşmezse öğrenme oranını (lr) factor kadar küçültür.
# Örn: patience=5 → 5 epoch iyileşme yoksa lr = lr * 0.5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ─────────────────────────────────────────────
# 9. Eğitim Döngüsü  (DataLoader + Gradient Clipping YENİ)
# ─────────────────────────────────────────────
num_epochs = 100
train_losses = []
val_losses   = []

# En iyi modeli takip etmek için  (DÜZELTME)
# ESKİ: model epoch 100'de kaydediliyordu — son epoch ≠ en iyi epoch ❌
# YENİ: val loss her iyileştiğinde model kaydedilir ✅
best_val_loss = float('inf')
MODEL_PATH    = "stock_model.pth"

for epoch in range(num_epochs):
    # ── Eğitim aşaması ──
    model.train()
    batch_losses = []

    for X_batch, y_batch in train_loader:     # DataLoader batch'leri döndürür
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()

        # Gradient Clipping (YENİ)
        # LSTM'lerde gradyanlar bazen çok büyüyebilir (exploding gradient).
        # max_norm=1.0: hiçbir gradyanın normu 1'i geçemez.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_losses.append(loss.item())

    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)

    # ── Validation aşaması ──
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            val_batch_losses.append(criterion(pred, y_batch).item())

    avg_val_loss = np.mean(val_batch_losses)
    val_losses.append(avg_val_loss)

    # En iyi modeli kaydet (val loss iyileşince)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

    # Scheduler'a validation loss'u ver
    scheduler.step(avg_val_loss)

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        best_marker = " ← best" if avg_val_loss == best_val_loss else ""
        print(f"Epoch [{epoch:>3}/{num_epochs-1}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}{best_marker}")

# ─────────────────────────────────────────────
# 10. En İyi Modeli Yükle  (DÜZELTME)
# ─────────────────────────────────────────────
# Eğitim boyunca en düşük val loss'a sahip ağırlıkları geri yükle.
# Böylece son epoch değil, gerçekten en iyi epoch'taki model kullanılır.
model.load_state_dict(torch.load(MODEL_PATH))
print(f"\nEn iyi model yüklendi → {MODEL_PATH}  (best val loss: {best_val_loss:.6f})")

# Sonradan başka bir yerde yüklemek için:
# model = PredictionModel().to(device)
# model.load_state_dict(torch.load(MODEL_PATH))

# ─────────────────────────────────────────────
# 11. Test ve Değerlendirme
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_test_pred  = model(X_test)
    y_train_pred = model(X_train.to(device))

y_train_pred_inv = target_scaler.inverse_transform(y_train_pred.cpu().numpy())
y_train_inv      = target_scaler.inverse_transform(y_train.numpy())

y_test_pred_inv  = target_scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_inv       = target_scaler.inverse_transform(y_test.cpu().numpy())

train_rmse = root_mean_squared_error(y_train_inv, y_train_pred_inv)
test_rmse  = root_mean_squared_error(y_test_inv,  y_test_pred_inv)

print(f"\nTrain RMSE: {train_rmse:.2f} $")
print(f"Test  RMSE: {test_rmse:.2f}  $")

# ─────────────────────────────────────────────
# 12. Gerçek vs Tahmin Tablosu
# ─────────────────────────────────────────────
# Test setindeki her gün için gerçek fiyat, tahmin ve hata yan yana
test_dates = df.index[-len(y_test_inv):]

results_df = pd.DataFrame({
    "Tarih"          : test_dates,
    "Gerçek Fiyat $" : y_test_inv.flatten().round(2),
    "Tahmin $"       : y_test_pred_inv.flatten().round(2),
    "Hata $"         : (y_test_inv.flatten() - y_test_pred_inv.flatten()).round(2),
    "Mutlak Hata $"  : np.abs(y_test_inv.flatten() - y_test_pred_inv.flatten()).round(2),
})
results_df = results_df.set_index("Tarih")

print("\n── Test Seti: Gerçek vs Tahmin (son 10 gün) ──")
print(results_df.tail(10).to_string())

# Tablonun tamamını CSV olarak da kaydedebilirsin:
# results_df.to_csv("tahminler.csv")

# ─────────────────────────────────────────────
# 13. Yarınki (Bir Sonraki Gün) Fiyat Tahmini
# ─────────────────────────────────────────────
# Elimizdeki en son 30 günü alıp modele veriyoruz.
# Model hiç görmediği bu pencereyle yarını tahmin eder.

last_30_features = feature_scaler.transform(
    df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-seq_length:]
)  # shape: (30, 5)

last_30_tensor = torch.tensor(last_30_features, dtype=torch.float32) \
                      .unsqueeze(0)          \
                      .to(device)            # shape: (1, 30, 5)

model.eval()
with torch.no_grad():
    next_day_scaled = model(last_30_tensor)

next_day_price = target_scaler.inverse_transform(
    next_day_scaled.cpu().numpy()
)[0, 0]

last_real_price = float(df['Close'].values[-1])

print(f"\n{'─'*40}")
print(f"  Son Gerçek Fiyat  : {last_real_price:.2f} $  ({df.index[-1].date()})")
print(f"  Tahmini Sonraki Gün Fiyatı : {next_day_price:.2f} $")
change = next_day_price - last_real_price
arrow = "▲" if change >= 0 else "▼"
print(f"  Beklenen Değişim  : {arrow} {abs(change):.2f} $ ({change/last_real_price*100:.2f}%)")
print(f"{'─'*40}\n")

# ─────────────────────────────────────────────
# 14. Görselleştirme
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = fig.add_gridspec(5, 1)

# — Fiyat Tahmini —
ax1 = fig.add_subplot(gs[:3, 0])
test_dates = df.index[-len(y_test_inv):]
ax1.plot(test_dates, y_test_inv,      color="blue",  label="Gerçek Fiyat")
ax1.plot(test_dates, y_test_pred_inv, color="green", label="Tahmin")
ax1.legend()
ax1.set_title(f"{ticker} Hisse Senedi Fiyat Tahmini (OHLCV | DataLoader | Grad Clip)", fontsize=14)
ax1.set_xlabel("Tarih")
ax1.set_ylabel("Fiyat ($)")
ax1.grid(True, alpha=0.3)

# — Hata Grafiği —
ax2 = fig.add_subplot(gs[3, 0])
prediction_error = np.abs(y_test_inv.flatten() - y_test_pred_inv.flatten())
ax2.axhline(test_rmse, color="blue", linestyle="--", label=f"Test RMSE: {test_rmse:.2f}$")
ax2.plot(test_dates, prediction_error, color="red", label="Mutlak Hata")
ax2.legend()
ax2.set_title("Tahmin Hatası ($)")
ax2.set_xlabel("Tarih")
ax2.set_ylabel("Hata ($)")
ax2.grid(True, alpha=0.3)

# — Train vs Validation Loss —  (YENİ — öğrenmeyi takip etmek için)
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
