# sequence_preparation.py
# 負責生成時間序列資料、分割訓練/驗證/測試集，並進行標準化。

# 導入必要的套件
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_multi_step_dataset(df, feature_cols, target_col='Log_Quantity', lookback=7, horizon=7):
    """
    建構多步預測的時間序列資料。

    參數：
        df (pd.DataFrame): 包含特徵和目標變量的資料
        feature_cols (list): 特徵欄位名稱
        target_col (str): 目標變量名稱，預設為 'Log_Quantity'
        lookback (int): 回顧天數，預設為7
        horizon (int): 預測天數，預設為7

    返回：
        X_seq (np.array): 特徵序列資料
        Y_seq (np.array): 目標序列資料
    """
    X_list, Y_list = [], []
    grouped = df.groupby('ProductID')
    for pid, subdf in grouped:
        subdf = subdf.sort_values('SalesDate')
        subX = subdf[feature_cols].values
        subY = subdf[target_col].values
        for i in range(len(subdf) - lookback - horizon + 1):
            X_seq = subX[i:i + lookback]
            y_seq = subY[i + lookback:i + lookback + horizon]
            X_list.append(X_seq)
            Y_list.append(y_seq)
    return np.array(X_list), np.array(Y_list)

def split_and_scale_data(main_df, feature_cols):
    """
    對資料進行時間序列切分和標準化。

    參數：
        main_df (pd.DataFrame): 包含特徵的資料
        feature_cols (list): 特徵欄位名稱

    返回：
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test (np.array): 標準化後的資料
        scaler (StandardScaler): 標準化器
    """
    # (D) 建構多步序列資料
    main_df.sort_values(['ProductID', 'SalesDate'], inplace=True)
    lookback = 7
    horizon = 7
    X_seq, Y_seq = create_multi_step_dataset(main_df, feature_cols, lookback=lookback, horizon=horizon)
    print("X_seq shape:", X_seq.shape)
    print("Y_seq shape:", Y_seq.shape)

    # (E) 時間序列切分
    N = len(X_seq)
    train_size = int(N * 0.7)
    val_size = int(N * 0.15)
    test_size = N - train_size - val_size

    X_train = X_seq[:train_size]
    y_train = Y_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    y_val = Y_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]
    y_test = Y_seq[train_size + val_size:]

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:  ", X_val.shape, y_val.shape)
    print("Test shape: ", X_test.shape, y_test.shape)

    # (F) 標準化
    scaler = StandardScaler()
    n_features = X_train.shape[2]

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler

# 測試程式碼
if __name__ == "__main__":
    import data_preprocessing as dp
    import feature_engineering as fe
    main_df, _ = dp.preprocess_data()
    main_df, feature_cols, _ = fe.engineer_features(main_df)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = split_and_scale_data(main_df, feature_cols)
    print("標準化完成")