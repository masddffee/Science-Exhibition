# model_training_xgb.py
# 負責XGBoost模型的訓練與評估，使用多輸出回歸進行7天預測。

# 導入必要的套件
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

def train_xgb_model(X_train_scaled, y_train, X_val_scaled, y_val):
    """
    訓練XGBoost模型，使用多輸出回歸和參數搜尋進行優化。

    參數：
        X_train_scaled (np.array): 訓練集特徵（標準化後）
        y_train (np.array): 訓練集目標
        X_val_scaled (np.array): 驗證集特徵（標準化後）
        y_val (np.array): 驗證集目標

    返回：
        best_multi_xgb (MultiOutputRegressor): 最佳XGBoost模型
        val_mse_xgb, val_r2_xgb, val_mae_xgb (float): 驗證集評估指標
        y_val_pred_xgb (np.array): 驗證集預測結果
    """
    # 將3D序列資料展平為2D，適合XGBoost
    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)

    # 計算樣本權重，對高銷量數據給予更高權重
    sample_weights = np.expm1(y_train) + 1
    mean_weight = np.mean(sample_weights, axis=0)
    sample_weights = np.where(sample_weights > mean_weight, sample_weights * 2, sample_weights)
    sample_weights = np.mean(sample_weights, axis=1)
    print("Sample weights shape:", sample_weights.shape)

    # 定義XGBoost模型
    base_xgb = XGBRegressor(objective='reg:squaredlogerror', random_state=42)
    multi_xgb = MultiOutputRegressor(base_xgb)

    # 定義參數搜尋範圍
    param_dist = {
        "estimator__n_estimators": [100, 200, 300, 500],
        "estimator__max_depth": [3, 5, 7, 9],
        "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "estimator__subsample": [0.6, 0.7, 0.8, 1.0],
        "estimator__colsample_bytree": [0.6, 0.7, 0.8, 1.0]
    }

    # 使用RandomizedSearchCV進行參數優化
    random_search = RandomizedSearchCV(
        estimator=multi_xgb,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        error_score='raise'
    )

    # 訓練模型
    random_search.fit(X_train_flat, y_train, sample_weight=sample_weights)
    print("XGB Best params:", random_search.best_params_)
    print("XGB Best score:", random_search.best_score_)

    # 使用最佳模型進行預測
    best_multi_xgb = random_search.best_estimator_
    y_val_pred_xgb_log = best_multi_xgb.predict(X_val_flat)
    y_val_pred_xgb = np.expm1(y_val_pred_xgb_log)
    y_val_original = np.expm1(y_val)

    # 計算驗證集評估指標
    val_mse_xgb = mean_squared_error(y_val_original.flatten(), y_val_pred_xgb.flatten())
    val_r2_xgb = r2_score(y_val_original.flatten(), y_val_pred_xgb.flatten())
    val_mae_xgb = mean_absolute_error(y_val_original.flatten(), y_val_pred_xgb.flatten())
    print("XGB Val MSE:", val_mse_xgb, "Val R2:", val_r2_xgb, "Val MAE:", val_mae_xgb)

    return best_multi_xgb, val_mse_xgb, val_r2_xgb, val_mae_xgb, y_val_pred_xgb

# 測試程式碼
if __name__ == "__main__":
    import data_preprocessing as dp
    import feature_engineering as fe
    import sequence_preparation as sp
    main_df, _ = dp.preprocess_data()
    main_df, feature_cols, _ = fe.engineer_features(main_df)
    X_train_scaled, y_train, X_val_scaled, y_val, _, _, _ = sp.split_and_scale_data(main_df, feature_cols)
    best_xgb, mse, r2, mae, _ = train_xgb_model(X_train_scaled, y_train, X_val_scaled, y_val)
    print("XGBoost訓練完成")