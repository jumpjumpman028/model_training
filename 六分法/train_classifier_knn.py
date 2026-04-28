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
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            
            word1 = data.get('word1')
            word2 = data.get('word2')
            label = data.get('category')
            
            if word1 and word2 and label is not None:
                X.append(extract_features(word1, word2, model))
                Y.append(label)
                
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
target_names = le.classes_ # 抓取類別名稱

# ==========================================
# 🚀 4. 設定要測試的進階參數組合
# ==========================================
k_values = [500]
metrics = ['cosine']
weights_options = ['distance']

# 用來收集所有「區塊」的 DataFrame 列表
all_reports_dfs = []

print("\n" + "="*60)
print("🔥 啟動 GPU KNN (直式區塊報表版) 🔥")
print("="*60)

total_combinations = len(k_values) * len(metrics) * len(weights_options)
current_run = 1

# ==========================================
# 🔄 5. 開始自動迴圈測試並建構 DataFrame 區塊
# ==========================================
for m in metrics:
    for w in weights_options:
        for k in k_values:
            print(f"\n▶️ [{current_run}/{total_combinations}] 測試組合 => Metric: {m} | Weights: {w} | K: {k}")
            
            clf = KNeighborsClassifier(n_neighbors=k, metric=m, weights=w, algorithm='brute')
            
            start_time = time.time()
            clf.fit(X_train_scaled, Y_train_encoded)
            y_pred = clf.predict(X_test_scaled)
            total_time = time.time() - start_time
            
            # 取得詳細報告
            report = classification_report(Y_test_encoded, y_pred, target_names=target_names, output_dict=True, zero_division=0)
            
            # --- 開始建立這個參數組合的專屬 DataFrame 區塊 ---
            
            # 第一段：參數與耗時
            df_params = pd.DataFrame([
                ['Metric', 'Weights', 'K Value', 'Time (s)'],
                [m, w, k, round(total_time, 2)]
            ])
            
            # 第二段：總體平均成績
            df_overall = pd.DataFrame([
                ['【總】Accuracy (%)', '【總】Macro Precision (%)', '【總】Macro Recall (%)', '【總】Macro F1 (%)'],
                [
                    round(report['accuracy'] * 100, 2),
                    round(report['macro avg']['precision'] * 100, 2),
                    round(report['macro avg']['recall'] * 100, 2),
                    round(report['macro avg']['f1-score'] * 100, 2)
                ]
            ])
            
            # 第三段：各 Level 詳細成績 (動態生成)
            level_data = []
            for cls_name in target_names:
                if cls_name in report:
                    # 第一列是標題
                    level_data.append([f"[{cls_name}] Precision (%)", f"[{cls_name}] Recall (%)", f"[{cls_name}] F1 (%)", ""])
                    # 第二列是數字
                    level_data.append([
                        round(report[cls_name]['precision'] * 100, 2),
                        round(report[cls_name]['recall'] * 100, 2),
                        round(report[cls_name]['f1-score'] * 100, 2),
                        "" # 第四格留空對齊
                    ])
            df_levels = pd.DataFrame(level_data)
            
            # 為了美觀，區塊之間加入空行
            df_empty = pd.DataFrame([["", "", "", ""]])
            
            # 將這個組合的所有小區塊拼接起來 (垂直合併)
            df_single_run = pd.concat([df_params, df_overall, df_empty, df_levels, df_empty, df_empty], ignore_index=True)
            
            # 存入總列表
            all_reports_dfs.append(df_single_run)
            
            print(f"   ┣ 耗時: {total_time:.2f} 秒 | 總 Macro F1: {report['macro avg']['f1-score']*100:.2f}%")
            current_run += 1

# ==========================================
# 💾 6. 將所有區塊匯出為 Excel
# ==========================================
# 把所有測試組合的 DataFrame 再垂直串接起來
final_df = pd.concat(all_reports_dfs, ignore_index=True)

excel_filename = os.path.join(current_dir, "knn_詳細正確率.xlsx")

# 這裡不寫入 header 和 index，因為我們已經在 DataFrame 裡面自己手刻標題列了
final_df.to_excel(excel_filename, index=False, header=False)

print("\n" + "="*60)
print(f"🎉 測試完畢！直式客製化報表已儲存至: {excel_filename}")
print("="*60)