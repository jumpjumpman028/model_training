import json
import numpy as np
import time
from gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors

# ================= 1. 路徑設定 =================
BIN_MODEL_PATH = 'cc.zh.300.bin'
KV_LOAD_PATH = 'fasttext_zh.kv'
JSON_TEST_PATH = 'testcase.json'  # 你可以改成 train.json 測更多

# ================= 2. 載入模型 =================
print("⏳ 正在載入模型...")
bin_model = load_facebook_model(BIN_MODEL_PATH)
kv_model = KeyedVectors.load(KV_LOAD_PATH, mmap='r')
print("✅ 模型載入完成！\n")
print("="*50)

# ================= 3. 從自訂 JSON 載入測試資 =================
JSON_TEST_PATH = 'testcase.json'
print(f"📂 正在讀取 {JSON_TEST_PATH} 並準備測試...")

words_list = []

with open(JSON_TEST_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
    # 遍歷四個分類，把裡面的字串全部塞進 list 裡
    for category, phrases in data.items():
        print(f"  - 載入 [{category}] 共 {len(phrases)} 筆")
        words_list.extend(phrases)

total_words = len(words_list)
print(f"🎯 成功載入 {total_words} 個測試案例！\n")

# ================= 4. 開始大規模自動比對 =================
print("🚀 開始進行 300 維向量比對...")
start_time = time.time()

match_count = 0
mismatch_count = 0
mismatch_words = [] # 用來記錄哪些詞彙出現差異

# 為了避免畫面被洗版，我們只顯示進度條或錯誤訊息
for i, word in enumerate(words_list):
    vec_bin = bin_model.wv[word]
    vec_kv = kv_model[word]
    
    # 比對兩個向量是否一致
    if np.allclose(vec_bin, vec_kv):
        match_count += 1
    else:
        mismatch_count += 1
        mismatch_words.append(word)
        
    # 每測完 100 個印一次進度，讓你知道程式沒當機
    if (i + 1) % 100 == 0:
        print(f"  ...已測試 {i + 1} / {total_words} 個詞彙")

print("="*50)

# ================= 5. 輸出統計報告 =================
print("\n📊 【大規模測試報告】")
print(f"總測試詞彙數: {total_words}")
print(f"🟢 完全一致數量: {match_count}")
print(f"🔴 出現差異數量: {mismatch_count}")
print(f"⏱️ 測試耗時: {time.time() - start_time:.2f} 秒")

# 如果真的有不一樣的，把它們印出來看看
if mismatch_count > 0:
    print("\n⚠️ 發現以下詞彙的向量有差異：")
    for w in mismatch_words[:10]: # 只印前10個避免洗版
        print(f" - {w}")
    if mismatch_count > 10:
        print(f" ...(還有 {mismatch_count - 10} 個)")
else:
    print("\n🎉 太完美了！所有的詞彙在兩個模型中輸出的向量 100% 相同！")