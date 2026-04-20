import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors

# 🚀 核心：使用 RAPIDS cuML 的 GPU KNN
from cuml.neighbors import KNeighborsClassifier

# ==========================================
# 📂 1. 設定路徑與載入模型
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
fasttext_model_path = os.path.join(current_dir, "fasttext_zh.kv")
train_file = os.path.join(current_dir, "train_data.jsonl")  
test_file = os.path.join(current_dir, "test_data.jsonl")

try:
    print("⏳ 正在極速載入 FastText .kv 模型...")
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
    print(f"📂 正在處理檔案: {file_path} ...")
    
    # 逐行讀取 JSONL，記憶體負擔極小
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳過空白行
                continue
                
            # 將這一行獨立解析為字典
            data = json.loads(line)
            
            # 提取特徵與標籤 (請確認你的 JSONL 中標籤的鍵值是 'category' 還是 'label')
            word1 = data.get('word1')
            word2 = data.get('word2')
            label = data.get('category') # 如果你的鍵名叫別的，請在這裡修改
            
            # 確保資料有成功抓到才進行萃取，避免報錯
            if word1 and word2 and label is not None:
                features = extract_features(word1, word2, model)
                X.append(features)
                Y.append(label)
                
    # 強制轉為 float32，GPU 運算最愛
    return np.array(X, dtype=np.float32), np.array(Y)

print("📂 正在讀取訓練集與測試集資料...")
X_train_raw, Y_train_raw = load_data(train_file, ft_model)
X_test_raw, Y_test_raw = load_data(test_file, ft_model)

# ==========================================
# ⚖️ 3. 資料前處理
# ==========================================
print("⚖️ 正在進行資料標準化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train_raw)
Y_test_encoded = le.transform(Y_test_raw)

# ==========================================
# 🚀 4. 設定要測試的進階參數組合
# ==========================================
# 1. K值 (鄰居數量)
k_values = [100,300]
# k_values = [5]
# 2. Metric (距離計算公式)
metrics = ['cosine']
# metrics = ['euclidean']
# 3. Weights (投票權重)
weights_options = ['uniform', 'distance']

results_list = []

print("\n" + "="*55)
print("🔥 啟動 GPU KNN 進階參數自動巡迴測試 (Grid Search) 🔥")
print("="*55)

# ==========================================
# 🔄 5. 開始自動迴圈測試 (共 24 種組合)
# ==========================================
total_combinations = len(k_values) * len(metrics) * len(weights_options)
current_run = 1

for m in metrics:
    for w in weights_options:
        for k in k_values:
            print(f"\n▶️ [{current_run}/{total_combinations}] 測試組合 => Metric: {m} | Weights: {w} | K: {k}")
            
            # 建立 KNN 模型 (強制使用 'brute' 確保顯卡最佳效能)
            clf = KNeighborsClassifier(
                n_neighbors=k, 
                metric=m, 
                weights=w,
                algorithm='brute' 
            )
            
            start_time = time.time()
            clf.fit(X_train_scaled, Y_train_encoded)
            
            # 預測
            print("   ⏳ 正在進行高維度距離計算預測...")
            y_pred = clf.predict(X_test_scaled)
            total_time = time.time() - start_time
            
            # 取得報告並提取四項指標
            report = classification_report(Y_test_encoded, y_pred, output_dict=True)
            acc = report['accuracy']
            mac_prec = report['macro avg']['precision']
            mac_rec = report['macro avg']['recall']
            mac_f1 = report['macro avg']['f1-score']
            
            # 印出這組的結果
            print(f"   ┣ 耗時: {total_time:.2f} 秒")
            print(f"   ┣ 1️⃣ 準確率 (Accuracy):      {acc*100:.2f}%")
            print(f"   ┣ 2️⃣ 巨觀精確率 (Macro Prec): {mac_prec*100:.2f}%")
            print(f"   ┣ 3️⃣ 巨觀召回率 (Macro Rec):  {mac_rec*100:.2f}%")
            print(f"   ┗ 4️⃣ 巨觀 F1 分數 (Macro F1):  {mac_f1*100:.2f}%")
            
            results_list.append({
                'Metric': m,
                'Weights': w,
                'K Value': k,
                'Accuracy (%)': round(acc * 100, 2),
                'Macro Precision (%)': round(mac_prec * 100, 2),
                'Macro Recall (%)': round(mac_rec * 100, 2),
                'Macro F1 (%)': round(mac_f1 * 100, 2)
            })
            current_run += 1

# ==========================================
# 💾 6. 將所有結果匯出為 Excel
# ==========================================
df_results = pd.DataFrame(results_list)

# 依據 Macro F1 分數由高到低排序，讓你一秒看出哪組最強
df_results = df_results.sort_values(by='Macro F1 (%)', ascending=False)

excel_filename = os.path.join(current_dir, "knn_advanced_tuning_results.xlsx")
df_results.to_excel(excel_filename, index=False)

print("\n" + "="*55)
print(f"🎉 24 種參數組合測試完畢！")
print(f"✅ 完整比較報表已自動排序並儲存至: {excel_filename}")
print("="*55)