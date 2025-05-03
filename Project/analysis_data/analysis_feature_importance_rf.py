# analysis_feature_importance_rf.py
# 使用隨機森林分析特徵重要性，並繪製圖表。

# 導入必要的套件
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def analyze_feature_importance_rf(input_path='data_with_weather_optimization.csv', output_path='outputs/feature_importance_rf.png'):
    """
    使用隨機森林分析特徵重要性，並繪製圖表。

    參數：
        input_path (str): 輸入資料檔案路徑，預設為 'data_with_weather_optimization.csv'
        output_path (str): 圖表輸出路徑，預設為 'outputs/feature_importance_rf.png'
    """
    # 讀取資料
    try:
        data = pd.read_csv(input_path, encoding="Big5")
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        raise

    # 定義特徵
    features = [
        "FixedPrice", "Month",
        "Weather_PCA1", "Weather_PCA2",
        "Lag_Temperature", "Lag_Precp", "Weather_Weekend",
        "Lag_1", "Lag_2", "Rolling_Mean_7", "Rolling_Sum_7"
    ]

    # 驗證特徵是否存在
    available_features = [col for col in features if col in data.columns]
    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        print(f"以下特徵在資料中不存在，將被忽略: {missing_features}")
    if not available_features:
        raise ValueError("沒有可用的特徵進行隨機森林分析")

    # 準備資料
    X = data[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(data['Quantity'], errors='coerce').fillna(0)

    # 訓練隨機森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 繪製特徵重要性
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(available_features, feature_importances, color='lightgreen')
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance from Random Forest")
    plt.tight_layout()

    # 儲存圖表
    try:
        plt.savefig(output_path)
        print(f"特徵重要性圖已儲存到 {output_path}")
    except Exception as e:
        print(f"儲存圖表失敗: {e}")
        raise
    plt.close()

# 測試程式碼
if __name__ == "__main__":
    analyze_feature_importance_rf()