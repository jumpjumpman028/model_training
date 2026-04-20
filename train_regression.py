import json
import numpy as np
import time
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from gensim.models import KeyedVectors

# 🚀 載入模型武器庫 (全 GPU 陣容)
# import xgboost as xgb
# from cuml.svm import SVR
from cuml.ensemble import RandomForestRegressor
# from cuml.linear_model import Ridge, ElasticNet

# ==========================================
# 📂 1. 設定分數對應表與路徑
# ==========================================
SCORE_MAPPING = {
    'level_0': 1.0, 'level_1': 0.92, 'level_2': 0.8,
    'level_3': 0.58, 'level_4': 0.29, 'level_5': 0.00
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
    X, Y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            
            score = None
            if 'score' in data:
                score = float(data['score'])
            elif 'category' in data and data['category'] in SCORE_MAPPING:
                score = SCORE_MAPPING[data['category']]
            
            if data.get('word1') and data.get('word2') and score is not None:
                X.append(extract_features(data['word1'], data['word2'], model))
                Y.append(score)
                
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n📦 讀取訓練集與測試集資料...")
X_train_raw, y_train = load_data(train_file, ft_model)
X_test_raw, y_test = load_data(test_file, ft_model)
print(f"📏 資料總覽: Train {X_train_raw.shape[0]} 筆, Test {X_test_raw.shape[0]} 筆")

# ==========================================
# ⚖️ 3. 資料標準化
# ==========================================
print("\n⚖️ 正在進行資料標準化 (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==========================================
# 🚀 4. 設定海量參數網格 (Grid Search 核心)
# ==========================================
# 定義每個模型要測試的參數字典
models_and_grids = [
    # {
    #     "name": "XGBoost (極限梯度提升)",
    #     "base_model": xgb.XGBRegressor(tree_method='hist', device='cuda', random_state=42),
    #     "param_grid": {
    #         'n_estimators': [300, 500, 800],   # 樹的數量：測試 3 種
    #         'max_depth': [6, 8, 10],           # 樹的深度：測試 3 種
    #         'learning_rate': [0.01, 0.05, 0.1] # 學習率：測試 3 種 (總共 3x3x3=27 組)
    #     }
    # },
    # {
    #     "name": "SVR (支持向量迴歸)",
    #     "base_model": SVR(kernel='rbf'),
    #     "param_grid": {
    #         'C': [0.1, 1.0, 5.0, 10.0],        # 懲罰係數：測試 4 種
    #         'gamma': ['scale', 'auto']         # 影響範圍：測試 2 種 (總共 4x2=8 組)
    #     }
    # },
    {
        "name": "RandomForest (隨機森林)",
        "base_model": RandomForestRegressor(random_state=42),
        "param_grid": {
            'n_estimators': [100, 300, 500],   # 樹的數量：測試 3 種
            'max_depth': [10, 16],             # 樹的深度：測試 2 種
            'max_features': [0.3, 0.5]         # 每次看多少比例的特徵：測試 2 種 (總共 3x2x2=12 組)
        }
    }
    # {
    #     "name": "Ridge (脊迴歸 - L2正則化)",
    #     "base_model": Ridge(),
    #     "param_grid": {
    #         'alpha': [0.1, 1.0, 10.0, 100.0]   # 正則化強度：測試 4 種 (總共 4 組)
    #     }
    # },
    # {
    #     "name": "ElasticNet (彈性網 - L1+L2)",
    #     "base_model": ElasticNet(),
    #     "param_grid": {
    #         'alpha': [0.1, 1.0],               # 整體強度：測試 2 種
    #         'l1_ratio': [0.2, 0.5, 0.8]        # L1與L2的比例：測試 3 種 (總共 2x3=6 組)
    #     }
    # }
]

# 展開所有的參數組合
all_tasks = []
for mg in models_and_grids:
    grid = ParameterGrid(mg["param_grid"])
    for params in grid:
        all_tasks.append({
            "name": mg["name"],
            "base_model": mg["base_model"],
            "params": params
        })

total_runs = len(all_tasks)
results_list = []

print("\n" + "="*65)
print(f"🔥 終極煉丹爐啟動：共計 {total_runs} 種模型參數組合 🔥")
print("="*65)

# ==========================================
# 🔄 5. 開始自動化大亂鬥
# ==========================================
for idx, task in enumerate(all_tasks, 1):
    model_name = task["name"]
    params = task["params"]
    
    print(f"\n▶️ [{idx}/{total_runs}] 正在訓練: {model_name}")
    print(f"   ┣ 參數: {params}")
    
    # 將這組參數設定給模型
    clf = task["base_model"]
    clf.set_params(**params)
    
    start_time = time.time()
    
    # 訓練模型
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # 預測與評估
    y_pred = clf.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   ┣ 耗時: {train_time:.2f} 秒")
    print(f"   ┣ 🎯 MAE:  {mae:.4f}")
    print(f"   ┗ 🏆 R²:   {r2:.4f}")
    
    # 儲存結果
    results_list.append({
        'Model': model_name,
        'Parameters': str(params),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2 Score': round(r2, 4),
        'Time (s)': round(train_time, 2)
    })
    
    # 💡 貼心設計：每跑完一種組合就「覆寫」一次 Excel，避免跑到一半當機心血全毀
    df_temp = pd.DataFrame(results_list).sort_values(by='MAE', ascending=True)
    df_temp.to_excel(os.path.join(current_dir, "massive_tuning_results_backup.xlsx"), index=False)

# ==========================================
# 💾 6. 輸出最終排行榜
# ==========================================
df_results = pd.DataFrame(results_list).sort_values(by='MAE', ascending=True)
excel_filename = os.path.join(current_dir, "ultimate_regression_leaderboard.xlsx")
df_results.to_excel(excel_filename, index=False)

print("\n" + "="*65)
print(f"🎉 煉丹完成！總共測試了 {total_runs} 種組合。")
print(f"✅ 最終戰力排行榜 (依 MAE 排名) 已儲存至: {excel_filename}")
print("="*65)