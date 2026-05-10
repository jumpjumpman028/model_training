import os
import glob
import json
import pandas as pd

# ==========================================
# 📂 1. 設定路徑 (預設為當前資料夾)
# ==========================================
TARGET_DIR = "./model_training/近義詞投票/data_set"

def generate_summary_table():
    # 尋找所有 result_ 開頭的 json 檔案
    json_files = glob.glob(os.path.join(TARGET_DIR, "result_*.json"))
    
    if not json_files:
        print("⚠️ 找不到任何 result_*.json 檔案！請確認路徑。")
        return

    print(f"📂 找到 {len(json_files)} 個 JSON 檔案，正在進行統計...\n")
    
    # 準備用來存放統計結果的列表
    summary_data = []

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        # 萃取出族名
        tribe_name = file_name.replace("result_", "").split("_")[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 初始化各分數的計數器
        score_counts = {
            "Level 0 (1.0)": 0,
            "Level 1 (0.8)": 0,
            "Level 2 (0.6)": 0,
            "Level 3 (0.4)": 0,
            "Level 4 (0.2)": 0,
            "都不是 (0.0)": 0,
            "總詞數": 0
        }
        
        for key, entry in data.items():
            if "分析結果" in entry:
                score = entry["分析結果"].get("判定分數", 0.0)
                
                if score == 1.0: score_counts["Level 0 (1.0)"] += 1
                elif score == 0.8: score_counts["Level 1 (0.8)"] += 1
                elif score == 0.6: score_counts["Level 2 (0.6)"] += 1
                elif score == 0.4: score_counts["Level 3 (0.4)"] += 1
                elif score == 0.2: score_counts["Level 4 (0.2)"] += 1
                else: score_counts["都不是 (0.0)"] += 1
                
                score_counts["總詞數"] += 1
                
        # 將族名加入字典
        score_counts["族語名稱"] = tribe_name
        summary_data.append(score_counts)

    # ==========================================
    # 📈 2. 建立原始 DataFrame
    # ==========================================
    df = pd.DataFrame(summary_data)
    score_cols = ["Level 0 (1.0)", "Level 1 (0.8)", "Level 2 (0.6)", "Level 3 (0.4)", "Level 4 (0.2)", "都不是 (0.0)"]
    cols = ["族語名稱"] + score_cols + ["總詞數"]
    df = df[cols]
    
    # 依照族語名稱排序
    df = df.sort_values(by="族語名稱").reset_index(drop=True)

    # ==========================================
    # 📈 3. 建立累計 DataFrame (由高分往低分累加)
    # ==========================================
    # 透過 Pandas 的 cumsum(axis=1) 進行橫向累加
    df_cum_values = df[score_cols].cumsum(axis=1)
    
    # 建立新的累計表格
    df_cumulative = pd.DataFrame({"族語名稱": df["族語名稱"]})
    for col in score_cols:
        # 修改欄位名稱加上「累計」
        df_cumulative[f"累計 {col}"] = df_cum_values[col]
    df_cumulative["總詞數"] = df["總詞數"]

    # ==========================================
    # 💾 4. 輸出結果 (包含雙分頁的 Excel)
    # ==========================================
    output_excel = os.path.join(TARGET_DIR, "16族同義詞判定分數統計表.xlsx")
    
    # 使用 ExcelWriter 建立兩個工作表 (Sheet)
    with pd.ExcelWriter(output_excel) as writer:
        df.to_excel(writer, sheet_name="各層級數量", index=False)
        df_cumulative.to_excel(writer, sheet_name="累計數量", index=False)
    
    print("="*60)
    print(f"🎉 統計完成！已處理 {len(json_files)} 個檔案。")
    print(f"👉 Excel 報表已儲存至: {output_excel}")
    print(f"   (💡 提示：打開 Excel 後，下方有「各層級數量」與「累計數量」兩個工作表可切換)")
    print("="*60)
    
    # 直接在終端機印出表格預覽 (改用 to_string 避免缺套件報錯)
    print("\n📊 【各層級數量】統計預覽：")
    print(df.to_string(index=False))

    print("\n📈 【累計數量 (由高至低)】統計預覽：")
    print(df_cumulative.to_string(index=False))

if __name__ == "__main__":
    generate_summary_table()