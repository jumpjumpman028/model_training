import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC  # 🌟 針對大數據建議使用 LinearSVC
from gensim.models import KeyedVectors

# ==========================================
# 📂 1. 設定與路徑
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
fasttext_model_path = os.path.join(current_dir, "fasttext_zh.kv")
train_file = os.path.join(current_dir, "train.jsonl")  
test_file = os.path.join(current_dir, "test.jsonl")

try:
    print("⏳ 正在極速載入 FastText 模型...")
    ft_model = KeyedVectors.load(fasttext_model_path, mmap='r')
except FileNotFoundError:
    print(f"❌ 找不到 {fasttext_model_path}，請確認檔案位置。")
    exit()

# ==========================================
# 🧠 2. 特徵萃取與資料載入
# ==========================================
def extract_features(word1, word2, model):
    vec1 = model[word1] if word1 in model else np.zeros(300)
    vec2 = model[word2] if word2 in model else np.zeros(300)
    cos_sim = model.similarity(word1, word2) if (word1 in model and word2 in model) else 0.0
    abs_diff = np.abs(vec1 - vec2)
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    return np.concatenate([vec1, vec2, abs_diff, np.array([cos_sim, euclidean_dist])])

def load_data(file_path, model):
    X, Y = [], []
    print(f"📂 正在逐行讀取 JSONL: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            word1, word2, label = data.get('word1'), data.get('word2'), data.get('label')
            if word1 and word2 and label in [0, 1]:
                X.append(extract_features(word1, word2, model))
                Y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

print("\n📦 讀取訓練集與測試集資料...")
X_train_raw, y_train = load_data(train_file, ft_model)
X_test_raw, y_test = load_data(test_file, ft_model)

# ==========================================
# ⚖️ 3. 資料標準化 (SVM 對縮放極度敏感，這步不可省略)
# ==========================================
print("\n⚖️ 正在進行資料標準化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==========================================
# 🚀 4. SVM 模型訓練
# ==========================================
# 🌟 使用 LinearSVC 並開啟平衡權重，以應對同義詞比例過低的問題
# C 參數控制懲罰力度，C 越大模型越不敢犯錯 (容易 Overfitting)
model = LinearSVC(
    C=5.0, 
    class_weight='balanced', 
    random_state=42, 
    
)

print(f"\n⚡ 啟動 SVM (LinearSVC) 分類訓練...")
start_time = time.time()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
elapsed = time.time() - start_time

# ==========================================
# 📊 5. 產出詳細診斷報表
# ==========================================
print("📝 正在產生詳細診斷報表...")

target_names = ['非同義詞 (0)', '同義詞 (1)']
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)

# 【區塊 A】: 總體成績
df_overall = pd.DataFrame([
    ['【總】Accuracy (準確率)', '【總】Macro Precision', '【總】Macro Recall', '【總】Macro F1', '耗時 (秒)'],
    [
        round(report['accuracy'] * 100, 2),
        round(report['macro avg']['precision'] * 100, 2),
        round(report['macro avg']['recall'] * 100, 2),
        round(report['macro avg']['f1-score'] * 100, 2),
        round(elapsed, 2)
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

# 【區塊 C】: 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
df_cm = pd.DataFrame([
    ['【實戰診斷】', '數量 (筆)', '代表意義', ''],
    ['成功抓出 (True Positive)', tp, '真的是同義詞，模型也答對', ''],
    ['漏網之魚 (False Negative)', fn, '明明是同義詞，模型卻漏抓', ''],
    ['成功排除 (True Negative)', tn, '不是同義詞，模型也沒被騙', ''],
    ['冤枉好人 (False Positive)', fp, '根本不是同義詞，模型卻亂抓', '']
])

df_empty = pd.DataFrame([[""] * 4])
final_report = pd.concat([df_overall, df_empty, df_levels, df_empty, df_cm], ignore_index=True)

excel_filename = os.path.join(current_dir, "svm_binary_classifier_report.xlsx")
final_report.to_excel(excel_filename, index=False, header=False)

print("\n" + "="*65)
print(f"🎉 SVM 分類測試完畢！\n👉 報表已儲存至: {excel_filename}")
print("="*65)