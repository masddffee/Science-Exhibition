# analysis_autocorrelation.py
# 負責分析銷售量的時間依賴性，繪製自相關圖。

# 導入必要的套件
import pandas as pd
import matplotlib.pyplot as plt

def analyze_autocorrelation(input_path='data_with_weather_optimization.csv', output_path='outputs/autocorrelation_plot.png'):
    """
    分析銷售量的時間依賴性，並繪製自相關圖。

    參數：
        input_path (str): 輸入資料檔案路徑，預設為 'data_with_weather_optimization.csv'
        output_path (str): 圖表輸出路徑，預設為 'outputs/autocorrelation_plot.png'
    """
    # 讀取資料
    try:
        data = pd.read_csv(input_path, encoding="Big5")
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        raise

    # 確保 Quantity 欄位存在且為數值型
    if 'Quantity' not in data.columns:
        raise ValueError("資料中缺少 'Quantity' 欄位")
    data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce').fillna(0)

    # 檢查滯後相關性
    lags = range(1, 31)  # 滯後 1 到 30 天
    autocorrelations = [data['Quantity'].autocorr(lag=lag) for lag in lags]

    # 繪製自相關圖
    plt.figure(figsize=(10, 6))
    plt.bar(lags, autocorrelations, color='skyblue')
    plt.xlabel("Lag (Days)")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of Sales Quantity")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 儲存圖表
    try:
        plt.savefig(output_path)
        print(f"自相關圖已儲存到 {output_path}")
    except Exception as e:
        print(f"儲存圖表失敗: {e}")
        raise
    plt.close()

# 測試程式碼
if __name__ == "__main__":
    analyze_autocorrelation()