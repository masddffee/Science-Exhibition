# evaluation_random_forest.py
# 負責隨機森林模型的訓練與評估，並儲存處理後的資料。

# 導入必要的套件
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def preprocess_and_evaluate_rf(
    X_train_path='X_train_time.csv',
    X_test_path='X_test_time.csv',
    y_train_path='y_train_time.csv',
    y_test_path='y_test_time.csv',
    output_X_train_path='outputs/X_train_time.csv',
    output_X_test_path='outputs/X_test_time.csv',
    output_y_train_path='outputs/y_train_time.csv',
    output_y_test_path='outputs/y_test_time.csv'
):
    """
    處理訓練與測試資料，訓練隨機森林模型並進行評估。

    參數：
        X_train_path, X_test_path, y_train_path, y_test_path (str): 輸入資料路徑
        output_X_train_path, output_X_test_path, output_y_train_path, output_y_test_path (str): 輸出資料路徑
    """
    # 讀取資料
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        raise

    # 確保 y_test 和 y_train 是一維數組
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # 處理日期類型的特徵
    if 'SalesDate' in X_train.columns:
        X_train['SalesDate'] = pd.to_datetime(X_train['SalesDate'], errors='coerce')
        X_test['SalesDate'] = pd.to_datetime(X_test['SalesDate'], errors='coerce')

        # 提取時間相關特徵
        for df in [X_train, X_test]:
            df['Year'] = df['SalesDate'].dt.year
            df['Month'] = df['SalesDate'].dt.month
            df['Day'] = df['SalesDate'].dt.day
            df['Weekday'] = df['SalesDate'].dt.weekday
            df.drop(columns=['SalesDate'], inplace=True)

    # 處理類別型字串特徵
    if 'ProductName' in X_train.columns:
        all_product_names = pd.concat([X_train['ProductName'], X_test['ProductName']]).unique()
        label_encoder = LabelEncoder()
        label_encoder.fit(all_product_names)
        X_train['ProductName'] = label_encoder.transform(X_train['ProductName'])
        X_test['ProductName'] = label_encoder.transform(X_test['ProductName'])

    if 'HolidayName' in X_train.columns:
        all_holiday_names = pd.concat([X_train['HolidayName'], X_test['HolidayName']]).unique()
        holiday_encoder = LabelEncoder()
        holiday_encoder.fit(all_holiday_names)
        X_train['HolidayName'] = holiday_encoder.transform(X_train['HolidayName'])
        X_test['HolidayName'] = holiday_encoder.transform(X_test['HolidayName'])

    # 填補缺失值
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # 保存處理後的資料
    try:
        X_train.to_csv(output_X_train_path, index=False)
        X_test.to_csv(output_X_test_path, index=False)
        pd.DataFrame(y_train, columns=['Quantity']).to_csv(output_y_train_path, index=False)
        pd.DataFrame(y_test, columns=['Quantity']).to_csv(output_y_test_path, index=False)
        print(f"處理後的資料已儲存到: {output_X_train_path}, {output_X_test_path}, {output_y_train_path}, {output_y_test_path}")
    except Exception as e:
        print(f"儲存資料失敗: {e}")
        raise

    # 訓練隨機森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 測試集評估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"隨機森林 MSE: {mse:.4f}")
    print(f"隨機森林 R² Score: {r2:.4f}")

    return model, mse, r2

# 測試程式碼
if __name__ == "__main__":
    preprocess_and_evaluate_rf()