import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gensim.models import KeyedVectors
import xgboost as xgb

# ==========================================
# 📂 1. 設定分數對應表與路徑 (🌟 已更新為真實的標籤名稱)
# ==========================================
SCORE_MAPPING = {
    'Synonym': 1.0, 
    'D1': 0.92, 
    'D2': 0.80,
    'D3': 0.58, 
    'D4': 0.29, 
    'D5': 0.00
}

current_dir = os.path.dirname(os.path.abspath(__file__))
fasttext_model_path = os.path.join(current_dir, "fasttext_zh.kv")
train_file = os.path.join(current_dir, "train_data.jsonl")  
test_file = os.path.join(current_dir, "test_data.jsonl")

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
    X, Y, Y_levels = [], [], []
    print(f"📂 正在逐行讀取 JSONL: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            
            category = data.get('category')
            word1 = data.get('word1')
            word2 = data.get('word2')
            
            # 🌟 只要你的 category 是 Synonym, D1, D2 等等，就會被收錄！
            if word1 and word2 and category in SCORE_MAPPING:
                X.append(extract_features(word1, word2, model))
                Y.append(SCORE_MAPPING[category])  # 迴歸分數 (例如 0.92)
                Y_levels.append(category)          # 用來畫報表的標籤 (例如 'D1')
                
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(Y_levels)

print("\n📦 讀取訓練集與測試集資料...")
X_train_raw, y_train, _ = load_data(train_file, ft_model)
X_test_raw, y_test, y_test_levels = load_data(test_file, ft_model)

print(f"📏 資料總覽: Train {X_train_raw.shape[0]} 筆, Test {X_test_raw.shape[0]} 筆")

# ==========================================
# ⚖️ 3. 資料標準化
# ==========================================
print("\n⚖️ 正在進行資料標準化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==========================================
# 🚀 4. XGBoost 模型訓練 (單次測試)
# ==========================================
params = {
    'learning_rate': 0.1,
    'max_depth': 10,
    'n_estimators': 500,
    'tree_method': 'hist',
    'device': 'cuda',      # 啟動 RTX 5070
    'random_state': 42
}

print(f"\n🔥 啟動 XGBoost 迴歸訓練: {params}")
model = xgb.XGBRegressor(**params)

start_time = time.time()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
elapsed = time.time() - start_time

# ==========================================
# 📊 5. 構建「總和＋拆解」的直式報表
# ==========================================
print("📝 正在產生詳細診斷報表...")

# 【區塊 A】: 參數紀錄
df_params = pd.DataFrame([
    ['Learning Rate', 'max_depth (深度)', 'n_estimators (樹數)', 'Time (s)'],
    [params['learning_rate'], params['max_depth'], params['n_estimators'], round(elapsed, 2)]
])

# 【區塊 B】: 總體成績
global_mae = mean_absolute_error(y_test, y_pred)
global_mse = mean_squared_error(y_test, y_pred)
global_rmse = np.sqrt(global_mse)
global_r2 = r2_score(y_test, y_pred)

df_overall = pd.DataFrame([
    ['【總】MAE', '【總】MSE', '【總】RMSE', '【總】R² Score'],
    [round(global_mae, 4), round(global_mse, 4), round(global_rmse, 4), round(global_r2, 4)]
])

# 【區塊 C】: 逐級拆解成績
level_rows = []

# 我們設定一個固定的顯示順序，讓報表看起來比較整齊
display_order = ['Synonym', 'D1', 'D2', 'D3', 'D4', 'D5']

for lvl in display_order:
    mask = (y_test_levels == lvl)
    
    if np.sum(mask) > 0:
        lvl_true = y_test[mask]
        lvl_pred = y_pred[mask]
        
        # 獨立計算該 Level 的指標
        lvl_mae = mean_absolute_error(lvl_true, lvl_pred)
        lvl_rmse = np.sqrt(mean_squared_error(lvl_true, lvl_pred))
        avg_pred = np.mean(lvl_pred)
        
        # 標題列
        level_rows.append([f"[{lvl}] 真實分數", f"[{lvl}] 平均預測值", f"[{lvl}] MAE 誤差", f"[{lvl}] RMSE"])
        # 數據列
        level_rows.append([
            SCORE_MAPPING[lvl], 
            round(avg_pred, 4), 
            round(lvl_mae, 4), 
            round(lvl_rmse, 4)
        ])

df_levels = pd.DataFrame(level_rows)

# 準備空行做分隔
df_empty = pd.DataFrame([[""] * 4])

# 將所有區塊垂直堆疊起來
final_report = pd.concat([
    df_params, 
    df_empty, 
    df_overall, 
    df_empty, 
    df_levels
], ignore_index=True)

# ==========================================
# 💾 6. 匯出 Excel
# ==========================================
excel_filename = os.path.join(current_dir, "xgb_regression_ultimate_report.xlsx")

# 關閉預設的 header 與 index，保留我們手刻的乾淨版面
final_report.to_excel(excel_filename, index=False, header=False)

print("\n" + "="*65)
print(f"🎉 測試完畢！包含【總和數據】與【Level 拆解數據】的報表已儲存至:\n👉 {excel_filename}")
print("="*65)