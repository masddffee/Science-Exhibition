# feature_engineering_lag.py
# 負責為資料添加滯後特徵和滾動特徵，並儲存更新後的資料。

# 導入必要的套件
import pandas as pd

def add_lag_and_rolling_features(input_path='data_with_weather_optimization.csv', output_path='data_with_weather_optimization.csv'):
    """
    為資料添加滯後特徵和滾動特徵，並儲存更新後的資料。

    參數：
        input_path (str): 輸入資料檔案路徑，預設為 'data_with_weather_optimization.csv'
        output_path (str): 輸出資料檔案路徑，預設為 'data_with_weather_optimization.csv'

    返回：
        data (pd.DataFrame): 更新後的資料
    """
    # 讀取資料，指定編碼為 Big5
    try:
        data = pd.read_csv(input_path, encoding="Big5")
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        raise

    # 確保 SalesDate 為日期格式並按日期排序
    data['SalesDate'] = pd.to_datetime(data['SalesDate'])
    data.sort_values(['SalesDate', 'ProductName'], inplace=True)

    # 添加滯後特徵（全局滯後）
    lags = [1, 2, 3, 4, 5, 7]
    for lag in lags:
        data[f'Lag_{lag}'] = data['Quantity'].shift(lag)

    # 添加滾動特徵（全局滾動）
    data['Rolling_Mean_7'] = data['Quantity'].rolling(window=7, min_periods=1).mean()
    data['Rolling_Sum_7'] = data['Quantity'].rolling(window=7, min_periods=1).sum()

    # 按商品分組添加滯後特徵
    data['Lag_1'] = data.groupby('ProductName')['Quantity'].shift(1)
    data['Lag_2'] = data.groupby('ProductName')['Quantity'].shift(2)

    # 填補缺失值（由於滯後和滾動特徵會導致首行數據為 NaN）
    for lag in lags:
        data[f'Lag_{lag}'].fillna(0, inplace=True)
    data['Rolling_Mean_7'].fillna(0, inplace=True)
    data['Rolling_Sum_7'].fillna(0, inplace=True)
    data['Lag_1'].fillna(0, inplace=True)
    data['Lag_2'].fillna(0, inplace=True)

    # 保存更新後的資料
    try:
        data.to_csv(output_path, index=False, encoding="Big5")
        print(f"已添加滯後特徵和滾動特徵，並保存為 {output_path}")
    except Exception as e:
        print(f"儲存資料失敗: {e}")
        raise

    return data

# 測試程式碼
if __name__ == "__main__":
    add_lag_and_rolling_features()