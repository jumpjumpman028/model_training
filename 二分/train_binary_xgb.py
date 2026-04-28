import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
import xgboost as xgb

# ==========================================
# 📂 1. 設定與路徑
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
fasttext_model_path = os.path.join(current_dir, "fasttext_zh.kv")
# ⚠️ 請確認你的檔名
train_file = os.path.join(current_dir, "train.jsonl")  
test_file = os.path.join(current_dir, "test.jsonl")

try:
    print("⏳ 正在極速載入 FastText 模型...")
    ft_model = KeyedVectors.load(fasttext_model_path, mmap='r')
except FileNotFoundError:
    print(f"❌ 找不到 {fasttext_model_path}，請確認檔案位置。")
    exit()

# ==========================================
# 🧠 2. 特徵萃取與資料載入 (專為 Binary JSONL 設計)
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
            
            word1 = data.get('word1')
            word2 = data.get('word2')
            label = data.get('label') # 🌟 直接抓取 0 或 1
            
            # 確保資料完整且 label 是 0 或 1
            if word1 and word2 and label in [0, 1]:
                X.append(extract_features(word1, word2, model))
                Y.append(int(label))
                
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

print("\n📦 讀取訓練集與測試集資料...")
X_train_raw, y_train = load_data(train_file, ft_model)
X_test_raw, y_test = load_data(test_file, ft_model)

print(f"📏 資料總覽: Train {X_train_raw.shape[0]} 筆, Test {X_test_raw.shape[0]} 筆")

# ==========================================
# ⚖️ 3. 資料標準化
# ==========================================
print("\n⚖️ 正在進行資料標準化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==========================================
# 🚀 4. XGBoost 二元分類模型訓練
# ==========================================
# 🌟 1. 先算出 0 和 1 的比例 (加在 params 上面)
num_zeros = np.sum(y_train == 0)
num_ones = np.sum(y_train == 1)
imbalance_ratio = num_zeros / num_ones if num_ones > 0 else 1.0
print(f"⚖️ 資料不平衡比例 (0的數量 / 1的數量) = {imbalance_ratio:.2f}")

# 🌟 2. 把這個比例塞進 XGBoost 的參數裡
params = {
    'learning_rate': 0.05,
    'max_depth': 10,
    'n_estimators': 500,
    'tree_method': 'hist',
    'device': 'cuda',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'scale_pos_weight': imbalance_ratio  # ⬅️ 加上這個救命神器！
}

print(f"\n🔥 啟動 XGBoost 二元分類訓練: {params}")
# 🌟 改用 XGBClassifier
model = xgb.XGBClassifier(**params)

start_time = time.time()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
elapsed = time.time() - start_time

# ==========================================
# 📊 5. 構建「二元分類專屬」直式報表
# ==========================================
print("📝 正在產生詳細診斷報表...")

# 【區塊 A】: 參數紀錄
df_params = pd.DataFrame([
    ['Learning Rate', 'max_depth (深度)', 'n_estimators (樹數)', 'Time (s)'],
    [params['learning_rate'], params['max_depth'], params['n_estimators'], round(elapsed, 2)]
])

# 【區塊 B】: 分類報告詳解
target_names = ['非同義詞 (0)', '同義詞 (1)']
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)

df_overall = pd.DataFrame([
    ['【總】Accuracy (準確率)', '【總】Macro Precision', '【總】Macro Recall', '【總】Macro F1'],
    [
        round(report['accuracy'] * 100, 2),
        round(report['macro avg']['precision'] * 100, 2),
        round(report['macro avg']['recall'] * 100, 2),
        round(report['macro avg']['f1-score'] * 100, 2)
    ]
])

# 【區塊 C】: 各類別明細
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

# 【區塊 D】: 🌟 混淆矩陣 (看抓到幾個、錯殺幾個)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

df_cm = pd.DataFrame([
    ['【實戰診斷】', '數量 (筆)', '代表意義', ''],
    ['成功抓出 (True Positive)', tp, '真的是同義詞，模型也答對', ''],
    ['漏網之魚 (False Negative)', fn, '明明是同義詞，模型卻沒看出來 (漏抓)', ''],
    ['成功排除 (True Negative)', tn, '不是同義詞，模型也沒被騙', ''],
    ['冤枉好人 (False Positive)', fp, '根本不是同義詞，模型卻亂抓', '']
])

# 將所有區塊垂直堆疊起來
df_empty = pd.DataFrame([[""] * 4])
final_report = pd.concat([
    df_params, df_empty, 
    df_overall, df_empty, 
    df_levels, df_empty, 
    df_cm
], ignore_index=True)

# ==========================================
# 💾 6. 匯出 Excel
# ==========================================
excel_filename = os.path.join(current_dir, "xgb_binary_classifier_詳細結果.xlsx")
final_report.to_excel(excel_filename, index=False, header=False)

print("\n" + "="*65)
print(f"🎉 二元分類測試完畢！包含【混淆矩陣實戰診斷】的報表已儲存至:\n👉 {excel_filename}")
print("="*65)