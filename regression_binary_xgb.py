import json
import numpy as np
import time
import pandas as pd
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    print(f"📂 正在讀取資料: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            # 迴歸任務直接讀取數值標籤 (不限於 0, 1)
            X.append(extract_features(data['word1'], data['word2'], model))
            Y.append(float(data['label'])) 
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

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
# 🚀 4. XGBoost Regression (GPU 版本)
# ==========================================
print(f"\n⚡ 啟動 XGBoost Regression GPU 加速訓練 (RTX 5070)...")

# 依照你提供的參數設定
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    tree_method='hist',
    device='cuda',
    random_state=42,
    verbosity=1
)

start_time = time.time()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
elapsed = time.time() - start_time

# ==========================================
# 📊 5. 計算指標並匯出 Excel
# ==========================================
print("📝 正在計算迴歸指標並產生 Excel 報表...")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 建立報表內容
df_report = pd.DataFrame([
    ['【迴歸指標】', '數值', '代表意義'],
    ['R² Score (決定係數)', f"{r2*100:.2f}%", '解釋變量佔比，越接近 100% 越好'],
    ['MSE (均方誤差)', round(mse, 6), '誤差平方的平均值，越小越好'],
    ['RMSE (根均方誤差)', round(rmse, 6), 'MSE 開根號，與標籤同單位'],
    ['MAE (平均絕對誤差)', round(mae, 6), '誤差絕對值的平均'],
    [''],
    ['【執行資訊】', '內容', ''],
    ['訓練耗時 (秒)', round(elapsed, 2), ''],
    ['使用裝置', 'NVIDIA GPU (CUDA)', ''],
    ['模型類型', 'XGBRegressor', '']
])

# 匯出檔案
excel_filename = os.path.join(current_dir, "xgb_regression_report.xlsx")
df_report.to_excel(excel_filename, index=False, header=False)

print("\n" + "="*60)
print(f"🎉 測試完成！R² 得分為: {r2*100:.2f}%")
print(f"👉 報表已儲存至: {excel_filename}")
print("="*60)