# feature_engineering.py
# 負責特徵工程：提取時間特徵、滯後特徵、交互項，並對目標變量進行對數轉換。

# 導入必要的套件
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(main_df):
    """
    對主力商品資料進行特徵工程，生成時間特徵、滯後特徵和交互項。

    參數：
        main_df (pd.DataFrame): 主力商品資料

    返回：
        main_df (pd.DataFrame): 包含新特徵的資料
        feature_cols (list): 特徵欄位名稱列表
        le_product (LabelEncoder): 商品名稱編碼器
    """
    # (C) 對主力商品做特徵工程
    # 提取日期相關特徵
    main_df['Year'] = main_df['SalesDate'].dt.year
    main_df['Month'] = main_df['SalesDate'].dt.month
    main_df['WeekOfYear'] = main_df['SalesDate'].dt.isocalendar().week.astype(int)
    main_df['DayOfWeek'] = main_df['SalesDate'].dt.weekday

    # 週期性編碼：將DayOfWeek轉為正弦和餘弦特徵，捕捉週期性
    main_df['DayOfWeek_sin'] = np.sin(2 * np.pi * main_df['DayOfWeek'] / 7)
    main_df['DayOfWeek_cos'] = np.cos(2 * np.pi * main_df['DayOfWeek'] / 7)

    # 編碼 ProductName 為數字
    le_product = LabelEncoder()
    main_df['ProductID'] = le_product.fit_transform(main_df['ProductName'])

    # 新增特徵工程
    # 1. 加入更長期的滾動統計（7天、14天、30天、60天）
    main_df['Rolling_Mean_14'] = main_df.groupby('ProductName')['Quantity'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    main_df['Rolling_Sum_14'] = main_df.groupby('ProductName')['Quantity'].transform(lambda x: x.rolling(14, min_periods=1).sum())
    main_df['Rolling_Mean_30'] = main_df.groupby('ProductName')['Quantity'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    main_df['Rolling_Sum_30'] = main_df.groupby('ProductName')['Quantity'].transform(lambda x: x.rolling(30, min_periods=1).sum())
    main_df['Rolling_Mean_60'] = main_df.groupby('ProductName')['Quantity'].transform(lambda x: x.rolling(60, min_periods=1).mean())

    # 2. 加入節日交互項
    main_df['Holiday_Product_Interaction'] = main_df['IsHoliday'] * main_df['ProductID']

    # 3. 加入產品與天氣交互項
    main_df['ProductID_Weather_Interaction'] = main_df['ProductID'] * (main_df['ExtremeTemperature'] + main_df['HeavyRain'])

    # 檢查核心欄位是否缺失
    required_cols = [
        'Lag_1', 'Lag_2', 'Rolling_Mean_7', 'Rolling_Sum_7',
        'FixedPrice', 'IsHoliday', 'IsWeekend', 'ExtremeTemperature', 'HeavyRain',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Month', 'WeekOfYear', 'Year'
    ]
    for c in required_cols:
        if c not in main_df.columns:
            print(f"[警告] 缺少欄位: {c}")

    # 定義特徵列表
    feature_cols = [
        'Lag_1', 'Lag_2', 'Rolling_Mean_7', 'Rolling_Sum_7', 'Rolling_Mean_14', 'Rolling_Sum_14',
        'Rolling_Mean_30', 'Rolling_Sum_30', 'Rolling_Mean_60',
        'FixedPrice', 'IsHoliday', 'IsWeekend', 'ExtremeTemperature', 'HeavyRain',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Month', 'WeekOfYear', 'Year', 'ProductID',
        'Holiday_Product_Interaction', 'ProductID_Weather_Interaction'
    ]

    # 對目標變量進行對數變換，處理偏態分佈
    main_df['Log_Quantity'] = np.log1p(main_df['Quantity'])

    return main_df, feature_cols, le_product

# 測試程式碼
if __name__ == "__main__":
    import data_preprocessing as dp
    main_df, _ = dp.preprocess_data()
    main_df, feature_cols, le_product = engineer_features(main_df)
    print("特徵欄位:", feature_cols)
    print("資料筆數:", len(main_df))