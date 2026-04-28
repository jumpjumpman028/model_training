import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from cuml.svm import SVC

current_dir = os.path.dirname(os.path.abspath(__file__))
print("⏳ [SVM] 正在極速載入 FastText .kv 模型...")
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

print("📂 [SVM] 讀取 JSONL 資料...")
X_train_raw, Y_train_raw = load_data(os.path.join(current_dir, "train_data.jsonl"), ft_model)
X_test_raw, Y_test_raw = load_data(os.path.join(current_dir, "test_data.jsonl"), ft_model)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train_raw)
Y_test_encoded = le.transform(Y_test_raw)

# 🚀 設定要測試的參數
C_values = [1.0, 5.0, 10.0, 30]
kernels = ['rbf', 'linear']

results_list = []
total_runs = len(C_values) * len(kernels)
current_run = 1

print("\n" + "="*50)
print("🔥 啟動 GPU SVM 參數自動巡迴測試 🔥")
print("="*50)

for k in kernels:
    for c in C_values:
        print(f"\n▶️ [{current_run}/{total_runs}] 測試組合 => Kernel: {k} | C值: {c}")
        
        clf = SVC(kernel=k, C=c, class_weight='balanced')
        start_time = time.time()
        clf.fit(X_train_scaled, Y_train_encoded)
        train_time = time.time() - start_time
        
        y_pred = clf.predict(X_test_scaled)
        report = classification_report(Y_test_encoded, y_pred, output_dict=True)
        
        acc, mac_prec, mac_rec, mac_f1 = report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']
        print(f"   ┣ 耗時: {train_time:.2f} 秒 | Macro F1: {mac_f1*100:.2f}%")
        
        results_list.append({'Kernel': k, 'C Value': c, 'Accuracy (%)': round(acc*100, 2), 'Macro Prec (%)': round(mac_prec*100, 2), 'Macro Rec (%)': round(mac_rec*100, 2), 'Macro F1 (%)': round(mac_f1*100, 2), 'Time (s)': round(train_time, 2)})
        current_run += 1

df_results = pd.DataFrame(results_list).sort_values(by='Macro F1 (%)', ascending=False)
df_results.to_excel(os.path.join(current_dir, "svm_grid_results.xlsx"), index=False)
print("\n✅ SVM 測試完畢！報表已儲存至 svm_grid_results.xlsx")