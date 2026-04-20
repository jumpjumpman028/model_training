import time
from gensim.models import KeyedVectors

# 指定剛剛轉換好的 .kv 檔案路徑
KV_LOAD_PATH = 'fasttext_zh.kv'

print("⚡ 開始快速載入模型...")
start_time = time.time()

# 關鍵技巧：加上 mmap='r' 
# 這會將龐大的矩陣映射到虛擬記憶體，不僅載入極快，還能大幅節省 RAM 的消耗
wv = KeyedVectors.load(KV_LOAD_PATH, mmap='r')

print(f"🚀 載入成功！總共耗時: {time.time() - start_time:.4f} 秒")
print("-" * 40)

# ==========================================
# 測試功能是否正常運作 (包含 OOV 未知詞推算)
# ==========================================

# 1. 找相似詞
print("【相似詞查詢】與「人工智慧」最相似的詞：")
for word, score in wv.most_similar('人工智慧', topn=3):
    print(f" - {word} ({score:.4f})")

# 2. 測試 FastText 的 OOV (未登錄詞) 推算能力
oov_word = "超酷炫機器人"
print(f"\n【未知詞測試】取得「{oov_word}」的向量：")
print(f"向量維度: {wv[oov_word].shape}")
