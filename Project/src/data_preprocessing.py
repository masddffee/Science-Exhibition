# data_preprocessing.py
# 負責資料的讀取、清理和冷門/主力商品分類，確保後續特徵工程和模型訓練有乾淨的資料。

# 導入必要的套件
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data_path='data_with_weather_optimization.csv'):
    """
    執行資料前處理：讀取資料、處理缺失值、分類冷門與主力商品。

    參數：
        data_path (str): 資料檔案路徑，預設為 'data_with_weather_optimization.csv'

    返回：
        main_df (pd.DataFrame): 主力商品資料
        cold_df (pd.DataFrame): 冷門商品資料
    """
    # (A) 讀取與基礎前處理
    # 讀取CSV檔案，並將缺失值填補為0
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)

    # 將 SalesDate 欄位轉為日期格式，並按日期和商品名稱排序
    data['SalesDate'] = pd.to_datetime(data['SalesDate'])
    data.sort_values(['SalesDate', 'ProductName'], inplace=True)

    # (B) 區分冷門商品與主力商品
    # 計算每個商品的資料筆數，設定閾值為30筆
    product_count_df = data.groupby('ProductName')['SalesDate'].count().reset_index()
    product_count_df.columns = ['ProductName', 'DataCount']
    threshold = 30
    cold_items = product_count_df[product_count_df['DataCount'] < threshold]['ProductName'].tolist()
    print("冷門商品列表:", cold_items)

    # 標記冷門商品（1為冷門，0為主力）
    data['IsCold'] = data['ProductName'].apply(lambda x: 1 if x in cold_items else 0)
    
    # 分割資料為主力商品和冷門商品
    main_df = data[data['IsCold'] == 0].copy()  # 主力商品資料
    cold_df = data[data['IsCold'] == 1].copy()  # 冷門商品資料

    return main_df, cold_df


if __name__ == "__main__":
    main_df, cold_df = preprocess_data()
    print("主力商品資料筆數:", len(main_df))
    print("冷門商品資料筆數:", len(cold_df))