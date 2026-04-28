import json
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
# 🌟 使用 NVIDIA cuML 的 GPU 版 KNN
from cuml.neighbors import KNeighborsClassifier 
from gensim.models import KeyedVectors

# ==========================================
# 📂 1. 設定路徑
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
    print(f"📂 正在讀取資料: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line.strip())
                # 🌟 只抓取 label 為 0 或 1 的資料，避免其他標籤導致報錯
                label = int(data.get('label', -1))
                if label in [0, 1]:
                    X.append(extract_features(data['word1'], data['word2'], model))
                    Y.append(label)
            except Exception:
                continue
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

print("\n📦 載入訓練集與測試集...")
X_train, y_train = load_data(train_file, ft_model)
X_test, y_test = load_data(test_file, ft_model)

# ==========================================
# ⚖️ 3. 資料標準化
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 🚀 4. GPU KNN 模型訓練與預測
# ==========================================
model = KNeighborsClassifier(
    n_neighbors=500, 
    weights='distance', 
    metric='cosine'
)

print(f"\n⚡ 啟動 GPU 加速 KNN (RTX 5070)...")
start_time = time.time()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
elapsed = time.time() - start_time

# ==========================================
# 📝 5. 直接輸出百分比格式文字報告
# ==========================================
# 取得 dict 格式的報告以便自定義格式
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*60)
print("🎯 KNN 分類器評估報告 (百分比格式)")
print("="*60)
print(f"🔹 模型參數: K=500, Weights=distance, Metric=cosine")
print(f"🔹 總預測耗時: {elapsed:.2f} 秒")
print("-" * 60)

# 總體指標
print(f"【總】Accuracy (準確率) : {report['accuracy']*100:.2f}%")
print(f"【總】Macro Precision    : {report['macro avg']['precision']*100:.2f}%")
print(f"【總】Macro Recall       : {report['macro avg']['recall']*100:.2f}%")
print(f"【總】Macro F1 Score     : {report['macro avg']['f1-score']*100:.2f}%")
print("-" * 60)

# 各類別細節
for label, name in [('0', '非同義詞 (0)'), ('1', '同義詞 (1)')]:
    metrics = report[label]
    print(f"📌 {name}:")
    print(f"   - Precision : {metrics['precision']*100:.2f}%")
    print(f"   - Recall    : {metrics['recall']*100:.2f}%")
    print(f"   - F1 Score  : {metrics['f1-score']*100:.2f}%")

print("-" * 60)
print(f"✅ 成功抓出 (TP): {tp:>6}  (真的是同義詞，模型也答對)")
print(f"❌ 漏網之魚 (FN): {fn:>6}  (明明是同義詞，模型卻漏抓)")
print(f"✅ 成功排除 (TN): {tn:>6}  (不是同義詞，模型也沒被騙)")
print(f"❌ 冤枉好人 (FP): {fp:>6}  (根本不是同義詞，模型卻亂抓)")
print("="*60)