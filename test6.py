
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

ticker = "MSFT"
df = yf.download(ticker, start="2020-01-01")

features     = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
close_prices = df['Close'].values.flatten()

direction = (close_prices[1:] > close_prices[:-1]).astype(np.float32)

up_ratio = direction.mean() * 100
print(f"Veri setinde günlerin %{up_ratio:.1f}'i yukarı kapandı.")
print(f"(Bu oran 'Her zaman YUKARI' baseline'ının tavanıdır)\n")

# ─────────────────────────────────────────────
# 4. Ölçeklendirme
# ─────────────────────────────────────────────
seq_length = 30
n = len(features)

train_end = int(0.70 * n)
val_end   = int(0.80 * n)

feature_scaler = MinMaxScaler()
feature_scaler.fit(features[:train_end])
features_scaled = feature_scaler.transform(features)

# ─────────────────────────────────────────────
# 5. Sequence Oluştur → Böl
# ─────────────────────────────────────────────
# Sequence i → [gün_i ... gün_{i+seq_len-1}] girdi
# Hedef      → direction[i+seq_len-1] (ertesi günün yönü)
def make_sequences(feat_scaled, dir_labels, seq_len):
    X, y = [], []
    for i in range(len(feat_scaled) - seq_len):
        X.append(feat_scaled[i : i + seq_len])
        y.append([dir_labels[i + seq_len - 1]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences(features_scaled, direction, seq_length)

train_seq_end = train_end - seq_length
val_seq_end   = val_end   - seq_length

X_train_np, y_train_np = X_all[:train_seq_end],            y_all[:train_seq_end]
X_val_np,   y_val_np   = X_all[train_seq_end:val_seq_end], y_all[train_seq_end:val_seq_end]
X_test_np,  y_test_np  = X_all[val_seq_end:],              y_all[val_seq_end:]

X_train = torch.tensor(X_train_np).to(device)
y_train = torch.tensor(y_train_np).to(device)
X_val   = torch.tensor(X_val_np).to(device)
y_val   = torch.tensor(y_val_np).to(device)
X_test  = torch.tensor(X_test_np).to(device)
y_test  = torch.tensor(y_test_np).to(device)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
print(f"Train seti yukarı oranı : %{y_train.mean().item()*100:.1f}")
print(f"Test  seti yukarı oranı : %{y_test.mean().item()*100:.1f}\n")

# ─────────────────────────────────────────────
# 6. Baseline Hesabı
# ─────────────────────────────────────────────
y_test_flat = y_test_np.flatten()

# Baseline 1: Her zaman YUKARI de
always_up_pred = np.ones_like(y_test_flat)

# Baseline 2: Dünkü yön (momentum)
yesterday_dir_pred = np.array([
    direction[val_seq_end + seq_length + i - 2]
    for i in range(len(y_test_flat))
])

baseline_always_up_acc = accuracy_score(y_test_flat, always_up_pred)
baseline_yesterday_acc = accuracy_score(y_test_flat, yesterday_dir_pred)

# ─────────────────────────────────────────────
# 7. Model  ★ MİMARİ DEĞİŞTİ ★
# ─────────────────────────────────────────────
class DirectionModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super(DirectionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        # Sigmoid YOK — BCEWithLogitsLoss içinde var (sayısal kararlılık için)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])   # raw logit döndür

model = DirectionModel().to(device)

# ─────────────────────────────────────────────
# 8. Loss, Optimizer, Scheduler
# ─────────────────────────────────────────────
# SORUN: Model "Her zaman YUKARI de" kestirmesini öğreniyor.
# Buna Majority Class Collapse deniyor — YUKARI zaten %53 doğru
# olduğu için model bu lokal minimumda takılıp kalıyor.
#
# ÇÖZÜM 1 — pos_weight: AŞAĞI'yı (0 sınıfı) yanlış tahmin etmenin
# cezasını artır. Böylece model her zaman YUKARI diyemez.
#   pos_weight = (AŞAĞI sayısı) / (YUKARI sayısı)
num_pos = y_train.sum().item()
num_neg = len(y_train) - num_pos
pos_weight_val = num_neg / num_pos
print(f"pos_weight: {pos_weight_val:.3f}  (YUKARI:{int(num_pos)}  AŞAĞI:{int(num_neg)})")

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_val]).to(device)
)

# ÇÖZÜM 2 — WeightedRandomSampler: Her batch'te YUKARI ve AŞAĞI
# günlerinden eşit sayıda örnek al. Model her iki sınıfı da eşit görür.
from torch.utils.data import WeightedRandomSampler

labels_np    = y_train_np.flatten()
class_counts = np.bincount(labels_np.astype(int))          # [AŞAĞI sayısı, YUKARI sayısı]
class_weights = 1.0 / class_counts                          # azınlık sınıfa daha yüksek ağırlık
sample_weights = class_weights[labels_np.astype(int)]       # her örneğin ağırlığı

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(sample_weights),
    replacement=True
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=7
)

# ─────────────────────────────────────────────
# 9. Eğitim
# ─────────────────────────────────────────────
BATCH_SIZE   = 64
# shuffle=False — sampler zaten karıştırıyor (ikisi birlikte kullanılamaz)
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

num_epochs     = 100
train_losses   = []
val_accuracies = []
best_val_acc   = 0.0
MODEL_PATH     = "direction_model.pth"

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())

    train_losses.append(np.mean(batch_losses))

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = (torch.sigmoid(model(X_batch)) >= 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    val_preds  = np.concatenate(all_preds).flatten()
    val_labels = np.concatenate(all_labels).flatten()
    val_acc    = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

    scheduler.step(val_acc)

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        marker = " ← best" if val_acc == best_val_acc else ""
        print(f"Epoch [{epoch:>3}/{num_epochs-1}] | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Acc: %{val_acc*100:.1f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")

# ─────────────────────────────────────────────
# 10. En İyi Modeli Yükle
# ─────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH))
print(f"\nEn iyi model yüklendi (best val acc: %{best_val_acc*100:.1f})")

# ─────────────────────────────────────────────
# 11. Test Değerlendirmesi
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs  = torch.sigmoid(test_logits).cpu().numpy().flatten()
    test_preds  = (test_probs >= 0.5).astype(float)

test_acc  = accuracy_score(y_test_flat,  test_preds)
test_prec = precision_score(y_test_flat, test_preds, zero_division=0)
test_rec  = recall_score(y_test_flat,    test_preds, zero_division=0)
test_f1   = f1_score(y_test_flat,        test_preds, zero_division=0)
cm        = confusion_matrix(y_test_flat, test_preds)

# ─────────────────────────────────────────────
# 12. Karşılaştırma Raporu
# ─────────────────────────────────────────────
def emoji(model_acc, ref_acc):
    return "✅ baseline'den iyi" if model_acc > ref_acc else "❌ baseline'den kötü!"

print(f"\n{'═'*58}")
print(f"  {'MODEL / BASELINE':<30} {'Accuracy':>10}  Sonuç")
print(f"{'─'*58}")
print(f"  {'Her Zaman YUKARI':<30} %{baseline_always_up_acc*100:>7.1f}  (referans)")
print(f"  {'Dünkü Yön (Momentum)':<30} %{baseline_yesterday_acc*100:>7.1f}  "
      f"{emoji(baseline_yesterday_acc, baseline_always_up_acc)}")
print(f"  {'LSTM Yön Modelimiz':<30} %{test_acc*100:>7.1f}  "
      f"{emoji(test_acc, baseline_always_up_acc)}")
print(f"{'─'*58}")
print(f"  Precision : %{test_prec*100:.1f}  (YUKARI dediğinde ne kadar doğru?)")
print(f"  Recall    : %{test_rec*100:.1f}  (Gerçek YUKARI günlerin kaçını yakaladı?)")
print(f"  F1 Score  : %{test_f1*100:.1f}")
print(f"{'═'*58}")

print(f"\n── Confusion Matrix ──")
print(f"                    Tahmin: AŞAĞI  Tahmin: YUKARI")
print(f"  Gerçek: AŞAĞI         {cm[0,0]:>5}          {cm[0,1]:>5}")
print(f"  Gerçek: YUKARI        {cm[1,0]:>5}          {cm[1,1]:>5}")

# ─────────────────────────────────────────────
# 13. Son 10 Gün Tablosu
# ─────────────────────────────────────────────
test_dates = df.index[seq_length + val_seq_end:]

results_df = pd.DataFrame({
    "Gerçek Yön"   : ["▲ YUKARI" if v == 1 else "▼ AŞAĞI" for v in y_test_flat],
    "LSTM Tahmini" : ["▲ YUKARI" if v == 1 else "▼ AŞAĞI" for v in test_preds],
    "Olasılık %"   : (test_probs * 100).round(1),
    "Doğru?"       : ["✅" if p == g else "❌" for p, g in zip(test_preds, y_test_flat)],
    "Naive Doğru?" : ["✅" if p == g else "❌" for p, g in zip(yesterday_dir_pred, y_test_flat)],
}, index=test_dates)

print(f"\n── Son 10 Gün Tahminleri ──")
print(results_df.tail(10).to_string())

# ─────────────────────────────────────────────
# 14. Yarınki Yön Tahmini
# ─────────────────────────────────────────────
last_30 = feature_scaler.transform(
    df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-seq_length:]
)
last_30_tensor = torch.tensor(last_30, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    next_prob = float(torch.sigmoid(model(last_30_tensor)).item())

next_dir = "▲ YUKARI" if next_prob >= 0.5 else "▼ AŞAĞI"
print(f"\n{'─'*40}")
print(f"  Son Kapanış         : {float(df['Close'].values[-1]):.2f} $  ({df.index[-1].date()})")
print(f"  Tahmini Yarınki Yön : {next_dir}")
print(f"  YUKARI olasılığı    : %{next_prob*100:.1f}")
print(f"  AŞAĞI olasılığı     : %{(1-next_prob)*100:.1f}")
print(f"{'─'*40}\n")

# ─────────────────────────────────────────────
# 15. Görselleştirme
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 13))
gs  = fig.add_gridspec(4, 2)

# — Confusion Matrix —
ax_cm = fig.add_subplot(gs[0, 0])
ax_cm.imshow(cm, cmap="Blues")
ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["AŞAĞI", "YUKARI"])
ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["AŞAĞI", "YUKARI"])
ax_cm.set_xlabel("Tahmin"); ax_cm.set_ylabel("Gerçek")
ax_cm.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

# — Accuracy Bar —
ax_bar = fig.add_subplot(gs[0, 1])
labels = ["Her Zaman\nYUKARI", "Dünkü\nYön", "LSTM\nModelimiz"]
accs   = [baseline_always_up_acc, baseline_yesterday_acc, test_acc]
colors = ["#ff9999", "#ffcc99",
          "#99cc99" if test_acc > baseline_always_up_acc else "#ff9999"]
bars   = ax_bar.bar(labels, [a*100 for a in accs], color=colors, edgecolor="black", width=0.5)
ax_bar.set_ylim(40, 70); ax_bar.set_ylabel("Accuracy (%)")
ax_bar.set_title("Model vs Baseline — Accuracy")
ax_bar.axhline(50, color="gray", linestyle="--", alpha=0.5, label="Rastgele (50%)")
for bar, acc in zip(bars, accs):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"%{acc*100:.1f}", ha="center", fontsize=11, fontweight="bold")
ax_bar.legend()

# — Tahmin Olasılıkları —
ax_prob = fig.add_subplot(gs[1, :])
ax_prob.plot(test_dates, test_probs, color="purple", lw=1, label="YUKARI olasılığı")
ax_prob.axhline(0.5, color="gray", linestyle="--", label="Karar sınırı (0.5)")
ax_prob.fill_between(test_dates, 0.5, test_probs,
                     where=(test_probs >= 0.5), alpha=0.2, color="green", label="YUKARI bölge")
ax_prob.fill_between(test_dates, test_probs, 0.5,
                     where=(test_probs < 0.5), alpha=0.2, color="red",   label="AŞAĞI bölge")
ax_prob.set_ylim(0, 1); ax_prob.legend(loc="upper right")
ax_prob.set_title("YUKARI Olasılıkları (Model Güveni)")
ax_prob.set_ylabel("Olasılık"); ax_prob.grid(True, alpha=0.3)

# — Doğru/Yanlış Noktalar —
ax_dots = fig.add_subplot(gs[2, :])
real_prices_test = close_prices[seq_length + val_seq_end:]
correct   = (test_preds == y_test_flat)
incorrect = ~correct
ax_dots.plot(test_dates, real_prices_test, color="blue", lw=1, alpha=0.5, label="Gerçek Fiyat")
ax_dots.scatter(test_dates[correct],   real_prices_test[correct],
                color="green", s=20, zorder=5, label=f"✅ Doğru ({correct.sum()})")
ax_dots.scatter(test_dates[incorrect], real_prices_test[incorrect],
                color="red",   s=20, zorder=5, label=f"❌ Yanlış ({incorrect.sum()})")
ax_dots.legend(); ax_dots.set_title("Fiyat Üzerinde Doğru/Yanlış Tahminler")
ax_dots.set_ylabel("Fiyat ($)"); ax_dots.grid(True, alpha=0.3)

# — Loss & Val Accuracy —
ax_loss  = fig.add_subplot(gs[3, :])
ax_loss_r = ax_loss.twinx()
ax_loss.plot(train_losses,     color="blue",   label="Train Loss (BCE)")
ax_loss_r.plot(val_accuracies, color="orange", label="Val Accuracy")
ax_loss_r.axhline(baseline_always_up_acc, color="gray", linestyle="--", alpha=0.7,
                   label=f"Baseline (%{baseline_always_up_acc*100:.1f})")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("BCE Loss", color="blue")
ax_loss_r.set_ylabel("Val Accuracy", color="orange")
ax_loss.set_title("Eğitim: Loss & Validation Accuracy")
lines1, lbl1 = ax_loss.get_legend_handles_labels()
lines2, lbl2 = ax_loss_r.get_legend_handles_labels()
ax_loss.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right")
ax_loss.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
