import json
import numpy as np
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import KeyedVectors


# ==========================================
# 📂 1. 設定與路徑
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
fasttext_model_path = os.path.join(current_dir, "fasttext_zh.kv")
train_file = os.path.join(current_dir,"task_01234_vs_5", "train.jsonl")  
test_file = os.path.join(current_dir,"task_01234_vs_5", "test.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用運算裝置: {device}")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight # 這裡會傳入自動算出的 torch.tensor
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 這裡套用 class_weights_tensor
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==========================================
# 🧠 2. 特徵萃取與資料載入
# ==========================================
def extract_features(word1, word2, model):
    vec1 = model[word1] if word1 in model else np.zeros(300)
    vec2 = model[word2] if word2 in model else np.zeros(300)
    cos_sim = model.similarity(word1, word2) if (word1 in model and word2 in model) else 0.0
    abs_diff = np.abs(vec1 - vec2)

    set1 = set(word1)
    set2 = set(word2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    char_overlap = intersection / union if union > 0 else 0.0

    return np.concatenate([vec1, vec2])

def load_data(file_path, model):
    X, Y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            if data.get('word1') and data.get('word2') and 'label' in data:
                X.append(extract_features(data['word1'], data['word2'], model))
                Y.append(int(data['label']))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

# 載入與標準化
ft_model = KeyedVectors.load(fasttext_model_path, mmap='r')
X_train_raw, y_train_raw = load_data(train_file, ft_model)
X_test_raw, y_test_raw = load_data(test_file, ft_model)

# 🌟 [功能寫入]：自動計算訓練資料比例與權重
print("\n📊 正在統計訓練資料比例...")
counts = np.bincount(y_train_raw)
n0, n1 = counts[0], counts[1]
w0 = 1.0
multiplier = 1.2 #調整1的權重
w1 = (n0 / n1) * multiplier  # 以數量多的類別為基準
class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float32).to(device)

print(f"   - 非近義詞 (0): {n0} 筆 (權重: {w0:.2f})")
print(f"   - 近義詞 (1)  : {n1} 筆 (權重: {w1:.2f})")
print(f"   - 套用 Tensor: torch.tensor([{w0:.2f}, {w1:.2f}])")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 🌟 20 萬筆資料，Batch Size 拉到 4096 速度最快
train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_raw)), batch_size=4096, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test_raw)), batch_size=4096, shuffle=False)

# ==========================================
# 🏗️ 3. 建立神經網路 (2 個輸出神經元)
# ==========================================
# 🌟 調整後的加寬版架構
class BinaryMLP(nn.Module):
    def __init__(self, input_dim):
        super(BinaryMLP, self).__init__()
        self.network = nn.Sequential(
            # 第一層直接拉高到 2048，給予足夠的空間捕捉 902 維特徵
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.1), # 使用 LeakyReLU 防止神經元壞死
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            
            # 最後輸出
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        return self.network(x)

# 🌟 既然模型變大了，權重衰減可以再稍微調高一點點，防止大模型亂跑


model = BinaryMLP(X_train_scaled.shape[1]).to(device)

# 🌟 換成 Focal Loss，針對少數樣本強力優化
criterion = FocalLoss(weight=class_weights_tensor, gamma=4.0)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# 🌟 學習率自動調速器：當 Test Loss 不再下降，自動把 LR 砍半
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# ==========================================
# 🔄 4. 訓練迴圈 (省略細節，維持原本邏輯)
# ==========================================
EPOCHS = 50
patience = 15
best_test_loss = float('inf')
epochs_no_improve = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(model(inputs), labels)
            test_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    test_loss /= len(test_loader.dataset)
    scheduler.step(test_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_binary_mlp.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience: break
# ==========================================
# 📊 5. 報表匯出 (完美融合版：總體指標 + 實戰診斷)
# ==========================================
print("\n📝 載入最佳權重，產生詳細診斷報表...")
# 加上 weights_only=True 是為了符合 PyTorch 最新版本的安全規範
model.load_state_dict(torch.load('best_binary_mlp.pth', weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        # 找出 2 個神經元中數值比較大的那一個的索引 (0 或 1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

y_pred = np.array(all_preds)

# 取得分類報告
target_names = ['非近義詞 (0)', '近義詞 (1)']
report = classification_report(y_test_raw, y_pred, target_names=target_names, output_dict=True, zero_division=0)

# 【區塊 A】: 總體成績 (你要的 Accuracy, Macro F1, Recall 全都在這！)
df_overall = pd.DataFrame([
    ['【總】Accuracy (準確率)', '【總】Macro Precision', '【總】Macro Recall', '【總】Macro F1'],
    [
        round(report['accuracy'] * 100, 2),
        round(report['macro avg']['precision'] * 100, 2),
        round(report['macro avg']['recall'] * 100, 2),
        round(report['macro avg']['f1-score'] * 100, 2)
    ]
])

# 【區塊 B】: 各類別明細
level_rows = []
for cls_name in target_names:
    level_rows.append([f"[{cls_name}] Precision (%)", f"[{cls_name}] Recall (%)", f"[{cls_name}] F1 (%)", ""])
    level_rows.append([
        round(report[cls_name]['precision'] * 100, 2),
        round(report[cls_name]['recall'] * 100, 2),
        round(report[cls_name]['f1-score'] * 100, 2),
        ""
    ])
df_levels = pd.DataFrame(level_rows)

# 【區塊 C】: 混淆矩陣實戰診斷
cm = confusion_matrix(y_test_raw, y_pred)
tn, fp, fn, tp = cm.ravel()
df_cm = pd.DataFrame([
    ['【實戰診斷】', '數量 (筆)', '代表意義', ''],
    ['成功抓出 (True Positive)', tp, '真的是同義詞，模型也答對', ''],
    ['漏網之魚 (False Negative)', fn, '明明是同義詞，模型卻漏抓', ''],
    ['成功排除 (True Negative)', tn, '不是同義詞，模型也沒被騙', ''],
    ['冤枉好人 (False Positive)', fp, '根本不是同義詞，模型卻亂抓', '']
])

# 將所有區塊垂直堆疊起來 (中間加入空行分隔)
df_empty = pd.DataFrame([[""] * 4])
final_report = pd.concat([
    df_overall, df_empty, 
    df_levels, df_empty, 
    df_cm
], ignore_index=True)

# 匯出 Excel
excel_filename = os.path.join(current_dir, "近義詞mlp_詳細正確率.xlsx")
final_report.to_excel(excel_filename, index=False, header=False)

total_train_time = time.time() - start_time
print("\n" + "="*60)
print(f"🎉 報表已生成！耗時 {total_train_time:.2f} 秒。請查看:\n👉 {excel_filename}")
print("="*60)