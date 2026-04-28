import json
import torch
import numpy as np
import time
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 🌟 修正後的匯入路徑
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 📂 1. 設定與路徑
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(current_dir, "train.jsonl")
test_file = os.path.join(current_dir, "test.jsonl")

# 參數設定
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 32           # 同義詞對通常很短，32 綽綽有餘
BATCH_SIZE = 32        # RTX 5070 (12GB) 跑 32 效能與速度最平衡
EPOCHS = 4             # BERT 微調通常 3~5 次即可收斂
LR = 2e-5              # 微調預訓練模型建議使用極小的學習率

# ==========================================
# 🧠 2. 資料集類別 (修正 encode_plus 問題)
# ==========================================
class SynonymDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = []
        print(f"📂 正在讀取資料: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                # 🌟 只取二元標籤 (0 或 1)
                label = int(item.get('label', -1))
                if label in [0, 1]:
                    self.data.append({
                        'text': (item['word1'], item['word2']),
                        'label': label
                    })
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        w1, w2 = self.data[item]['text']
        label = self.data[item]['label']

        # 🌟 修正處：直接呼叫 tokenizer 物件 (取代 encode_plus)
        encoding = self.tokenizer(
            w1, w2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 🏗️ 3. 初始化模型與權重 (修正 AdamW 參數)
# ==========================================
print(f"⏳ 正在下載/載入 {MODEL_NAME} 預訓練權重...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

train_dataset = SynonymDataset(train_file, tokenizer, MAX_LEN)
test_dataset = SynonymDataset(test_file, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 🌟 自動計算訓練資料比例 (權重補償)
labels_train = [d['label'] for d in train_dataset.data]
n0, n1 = np.bincount(labels_train)
weights = torch.tensor([1.0, n0/n1], dtype=torch.float32).to(device)
print(f"📊 訓練集統計 - 0: {n0} 筆, 1: {n1} 筆 | 加權比例: {n0/n1:.2f}")

# 🌟 修正處：使用 torch.optim.AdamW 並移除 correct_bias
optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights).to(device)

# ==========================================
# 🔄 4. 訓練流程
# ==========================================
print(f"🔥 開始在 {device} 上進行 BERT 微調...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 防止梯度爆炸
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    print(f"✅ Epoch {epoch+1}/{EPOCHS} 完成 | 平均 Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 📊 5. 評估與文字報告 (百分比格式)
# ==========================================
print("\n📝 訓練結束，正在進行測試集評估...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

elapsed = time.time() - start_time
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

print("\n" + "="*60)
print(f"🎯 BERT 同義詞辨識報告 (微調自: {MODEL_NAME})")
print("="*60)
print(f"【總】Accuracy (準確率) : {report['accuracy']*100:.2f}%")
print(f"【總】Macro Precision    : {report['macro avg']['precision']*100:.2f}%")
print(f"【總】Macro Recall       : {report['macro avg']['recall']*100:.2f}%")
print(f"【總】Macro F1 Score     : {report['macro avg']['f1-score']*100:.2f}%")
print(f"🔹 總耗時: {elapsed:.2f} 秒")
print("-" * 60)
print(f"📌 [同義詞 (1)] 表現明細:")
print(f"   - Precision (精確率) : {report['1']['precision']*100:.2f}%")
print(f"   - Recall    (召回率) : {report['1']['recall']*100:.2f}%")
print(f"   - F1 Score  (平衡分) : {report['1']['f1-score']*100:.2f}%")
print("-" * 60)
print(f"✅ 成功抓出 (TP): {tp:>6}  |  ❌ 漏網之魚 (FN): {fn:>6}")
print(f"✅ 成功排除 (TN): {tn:>6}  |  ❌ 冤枉好人 (FP): {fp:>6}")
print("="*60)

# 如果你需要儲存模型，可以解開下行註解
# torch.save(model.state_dict(), 'bert_synonym_model.pth')