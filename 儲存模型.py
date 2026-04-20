from gensim.models.fasttext import load_facebook_model

# 1. 指定你解壓縮出來的原始檔案路徑
BIN_MODEL_PATH = 'cc.zh.300.bin'
# 2. 指定轉換後的新檔案名稱
KV_SAVE_PATH = 'fasttext_zh.kv'

print("⏳ [一次性工作] 正在載入龐大的 .bin 模型，這需要幾分鐘，請耐心等候...")
model = load_facebook_model(BIN_MODEL_PATH)

print("💾 正在將模型轉換並儲存為輕量化的 KeyedVectors 格式...")
# 透過 model.wv.save 只保留推算詞向量與子詞所需的結構，捨棄訓練用的權重
model.wv.save(KV_SAVE_PATH)

print("✅ 轉換完成！以後你的主程式再也不用讀取那個 7GB 的 .bin 檔了！")
