# feature_engineering_weather_pca.py
# 負責天氣特徵的 PCA 處理、衍生特徵生成，並儲存更新後的資料。

# 導入必要的套件
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def process_weather_features(input_path='data.csv', output_path='data_with_weather_optimization.csv'):
    """
    對天氣特徵進行 PCA 處理，生成衍生特徵，並儲存更新後的資料。

    參數：
        input_path (str): 輸入資料檔案路徑，預設為 'data.csv'
        output_path (str): 輸出資料檔案路徑，預設為 'data_with_weather_optimization.csv'

    返回：
        data (pd.DataFrame): 更新後的資料
    """
    # 1. 讀取資料
    try:
        data = pd.read_csv(input_path, encoding="Big5")
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        raise

    # 2. 數據清理
    # 用 NaN 替換非數值型的字串
    data = data.replace('X', pd.NA)
    data = data.replace('T', pd.NA)

    # 轉換天氣欄位為數值類型
    weather_columns = ['Temperature', 'RH', 'WS', 'Precp', 'SunShine']
    for col in weather_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(data[col].mean())  # 填補缺失值
        else:
            print(f"警告: 欄位 {col} 不存在於資料中")

    # 3. 日期處理
    data['SalesDate'] = pd.to_datetime(data['SalesDate'], format='%Y/%m/%d', errors='coerce')
    data['Month'] = data['SalesDate'].dt.month

    # 4. 處理 'IsHoliday' 和 'IsWeekend'，轉換為 0 或 1
    if 'IsHoliday' in data.columns:
        data['IsHoliday'] = data['IsHoliday'].astype(int)
    if 'IsWeekend' in data.columns:
        data['IsWeekend'] = data['IsWeekend'].astype(int)

    # 5. 主成分分析 (PCA)
    # 選擇天氣相關特徵
    available_weather_features = [col for col in weather_columns if col in data.columns]
    if not available_weather_features:
        raise ValueError("沒有可用的天氣特徵進行 PCA 分析")

    scaler = StandardScaler()
    scaled_weather = scaler.fit_transform(data[available_weather_features])

    pca = PCA(n_components=2)  # 提取兩個主成分
    pca_weather = pca.fit_transform(scaled_weather)

    # 將 PCA 結果添加到資料中
    data['Weather_PCA1'] = pca_weather[:, 0]
    data['Weather_PCA2'] = pca_weather[:, 1]

    # 6. 增加極端天氣特徵
    if 'Temperature' in data.columns:
        data['ExtremeTemperature'] = data['Temperature'].apply(lambda x: 1 if x > 30 or x < 5 else 0)
    if 'Precp' in data.columns:
        data['HeavyRain'] = data['Precp'].apply(lambda x: 1 if x > 10 else 0)

    # 7. 滯後天氣特徵
    if 'Temperature' in data.columns:
        data['Lag_Temperature'] = data['Temperature'].shift(1)
    if 'Precp' in data.columns:
        data['Lag_Precp'] = data['Precp'].shift(1)

    # 8. 計算天氣與特定日期的交互影響
    if 'IsWeekend' in data.columns and 'Temperature' in data.columns:
        data['Weather_Weekend'] = data['IsWeekend'] * data['Temperature']
    if 'IsHoliday' in data.columns and 'Precp' in data.columns:
        data['Weather_Holiday'] = data['IsHoliday'] * data['Precp']

    # 9. 對某些天氣特徵進行對數變換（非線性影響）
    if 'Precp' in data.columns:
        data['Log_Precp'] = np.log1p(data['Precp'])  # log(Precp + 1) 避免 log(0)

    # 10. 分段處理溫度
    if 'Temperature' in data.columns:
        data['Temperature_Bin'] = pd.cut(data['Temperature'], bins=[-10, 0, 15, 25, 35, 50], labels=[1, 2, 3, 4, 5])

    # 11. 填補因滯後特徵產生的缺失值
    data.fillna(0, inplace=True)

    # 12. 保存更新後的資料
    try:
        data.to_csv(output_path, index=False, encoding="Big5")
        print(f"天氣特徵優化結果已保存到 {output_path}")
    except Exception as e:
        print(f"儲存資料失敗: {e}")
        raise

    return data

# 測試程式碼
if __name__ == "__main__":
    process_weather_features()