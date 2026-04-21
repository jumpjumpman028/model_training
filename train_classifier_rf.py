import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import KeyedVectors
from cuml.ensemble import RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
print("⏳ [RF] 正在極速載入 FastText .kv 模型...")
ft_model = KeyedVectors.load(os.path.join(current_dir, "fasttext_zh.kv"), mmap='r')

def extract_features(word1, word2, model):
    vec1 = model[word1] if word1 in model else np.zeros(300)
    vec2 = model[word2] if word2 in model else np.zeros(300)
    cos_sim = model.similarity(word1, word2) if (word1 in model and word2 in model) else 0.0
    abs_diff = np.abs(vec1 - vec2)
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    return np.concatenate([vec1, vec2, abs_diff, np.array([cos_sim, euclidean_dist])])

def load_data(file_path, model):
    X, Y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            if data.get('word1') and data.get('word2') and data.get('category') is not None:
                X.append(extract_features(data['word1'], data['word2'], model))
                Y.append(data['category'])
    return np.array(X, dtype=np.float32), np.array(Y)

print("📂 [RF] 讀取 JSONL 資料...")
X_train_raw, Y_train_raw = load_data(os.path.join(current_dir, "train_data.jsonl"), ft_model)
X_test_raw, Y_test_raw = load_data(os.path.join(current_dir, "test_data.jsonl"), ft_model)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 使用 LabelEncoder 並儲存類別名稱 (例如 'level_0', 'level_1')
le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train_raw)
Y_test_encoded = le.transform(Y_test_raw)
target_names = le.classes_  # 這裡會抓到所有標籤的真實名稱

# 🚀 設定要測試的參數
n_estimators_list = [500]
max_depth_list = [10]

results_list = []
total_runs = len(n_estimators_list) * len(max_depth_list)
current_run = 1

print("\n" + "="*60)
print("🔥 啟動 GPU 隨機森林參數自動巡迴測試 (含各等級詳解) 🔥")
print("="*60)

for trees in n_estimators_list:
    for depth in max_depth_list:
        print(f"\n▶️ [{current_run}/{total_runs}] 測試組合 => 樹木數量: {trees} | 最大深度: {depth}")
        
        clf = RandomForestClassifier(n_estimators=trees, max_depth=depth, max_features=0.5)
        start_time = time.time()
        clf.fit(X_train_scaled, Y_train_encoded)
        train_time = time.time() - start_time
        
        y_pred = clf.predict(X_test_scaled)
        
        # --- 取得詳細報表 ---
        # 為了能正確抓出各 Level 的分數，我們把 output_dict 打開
        report = classification_report(Y_test_encoded, y_pred, target_names=target_names, output_dict=True)
        
        # 1. 先把基本盤(總成績)存起來
        acc = report['accuracy']
        mac_prec = report['macro avg']['precision']
        mac_rec = report['macro avg']['recall']
        mac_f1 = report['macro avg']['f1-score']
        
        print(f"   ┣ 耗時: {train_time:.2f} 秒 | 總 Accuracy: {acc*100:.2f}% | 總 Macro F1: {mac_f1*100:.2f}%")
        
        # 準備要存入 Excel 的字典
        row_data = {
            'n_estimators (樹數)': trees, 
            'max_depth (深度)': depth, 
            'Time (s)': round(train_time, 2),
            '【總】Accuracy (%)': round(acc*100, 2), 
            '【總】Macro Prec (%)': round(mac_prec*100, 2), 
            '【總】Macro Rec (%)': round(mac_rec*100, 2), 
            '【總】Macro F1 (%)': round(mac_f1*100, 2)
        }
        
        # 2. 🌟 關鍵魔法：把每一個 Level 的成績獨立抽出來，塞進 Excel 的欄位裡
        # target_names 通常是 ['level_0', 'level_1', 'level_2'...]
        for cls_name in target_names:
            if cls_name in report:
                row_data[f"[{cls_name}] Prec (%)"] = round(report[cls_name]['precision'] * 100, 2)
                row_data[f"[{cls_name}] Rec (%)"]  = round(report[cls_name]['recall'] * 100, 2)
                row_data[f"[{cls_name}] F1 (%)"]   = round(report[cls_name]['f1-score'] * 100, 2)
        
        results_list.append(row_data)
        current_run += 1

# ==========================================
# 💾 將所有詳細結果匯出為 Excel
# ==========================================
df_results = pd.DataFrame(results_list)

# 依據總體 Macro F1 分數由高到低排序
df_results = df_results.sort_values(by='【總】Macro F1 (%)', ascending=False)

excel_filename = os.path.join(current_dir, "rf_詳細正確率.xlsx")
df_results.to_excel(excel_filename, index=False)

print("\n" + "="*60)
print(f"✅ RF 測試完畢！包含各等級詳細數據的報表已儲存至: {excel_filename}")
print("="*60)