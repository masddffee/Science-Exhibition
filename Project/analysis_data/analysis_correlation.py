# analysis_correlation.py
# 負責計算皮爾森相關性並繪製熱力圖。

# 導入必要的套件
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_correlation(input_path='data_with_weather_optimization.csv', output_path='outputs/correlation_matrix.png'):
    """
    計算皮爾森相關性並繪製熱力圖，分析特徵與銷量之間的關係。

    參數：
        input_path (str): 輸入資料檔案路徑，預設為 'data_with_weather_optimization.csv'
        output_path (str): 圖表輸出路徑，預設為 'outputs/correlation_matrix.png'
    """
    # 讀取資料
    try:
        data = pd.read_csv(input_path, encoding="Big5")
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        raise

    # 定義要分析的特徵
    selected_features = [
        'Quantity', "FixedPrice", "Temperature", "RH", "WS", "Precp", "SunShine", "Month",
        "Weather_PCA1", "Weather_PCA2", "ExtremeTemperature", "HeavyRain",
        "Lag_Temperature", "Lag_Precp", "Weather_Weekend", "Weather_Holiday",
        "Log_Precp", "Temperature_Bin",
        "Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5", "Lag_7",
        "Rolling_Mean_7", "Rolling_Sum_7"
    ]

    # 驗證選定欄位是否存在
    available_features = [col for col in selected_features if col in data.columns]
    missing_columns = [col for col in selected_features if col not in data.columns]
    if missing_columns:
        print(f"以下欄位在資料中不存在，將被忽略: {missing_columns}")
    if not available_features:
        raise ValueError("沒有可用的特徵進行相關性分析")

    # 提取數值資料並處理缺失值
    numeric_data = data[available_features].apply(pd.to_numeric, errors='coerce')
    numeric_data = numeric_data.fillna(0)

    # 計算相關矩陣
    correlation_matrix = numeric_data.corr()

    # 可視化相關矩陣
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix with Quantity')
    plt.tight_layout()

    # 儲存圖表
    try:
        plt.savefig(output_path)
        print(f"相關性熱力圖已儲存到 {output_path}")
    except Exception as e:
        print(f"儲存圖表失敗: {e}")
        raise
    plt.close()

    # 顯示 Quantity 與其他特徵的相關性
    if 'Quantity' in correlation_matrix.columns:
        quantity_correlation = correlation_matrix['Quantity'].sort_values(ascending=False)
        print("Quantity 與各特徵的相關性:")
        print(quantity_correlation)
    else:
        print("警告: 無法計算 Quantity 的相關性，可能是因為資料中缺少 Quantity 欄位")

# 測試程式碼
if __name__ == "__main__":
    analyze_correlation()