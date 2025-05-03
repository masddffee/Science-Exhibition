# model_ensemble.py
# 負責模型集成（動態權重融合）、冷門商品預測和最終預測結果生成。

# 導入必要的套件
import numpy as np
import pandas as pd
import joblib

def calculate_dynamic_weights(mse_xgb, mse_lstm, r2_xgb, r2_lstm, mae_xgb, mae_lstm, y_val_pred_xgb, y_val_pred_lstm):
    """
    計算XGBoost和LSTM的動態權重，基於誤差指標和穩定性。

    參數：
        mse_xgb, mse_lstm (float): XGBoost和LSTM的均方誤差
        r2_xgb, r2_lstm (float): XGBoost和LSTM的R2分數
        mae_xgb, mae_lstm (float): XGBoost和LSTM的平均絕對誤差
        y_val_pred_xgb, y_val_pred_lstm (np.array): 驗證集預測結果

    返回：
        alpha_xgb, alpha_lstm (float): 動態權重
    """
    mse_weight_xgb = 1 / mse_xgb
    mse_weight_lstm = 1 / mse_lstm
    mae_weight_xgb = 1 / mae_xgb
    mae_weight_lstm = 1 / mae_lstm
    r2_weight_xgb = r2_xgb
    r2_weight_lstm = r2_lstm
    pred_var_xgb = np.var(y_val_pred_xgb.flatten())
    pred_var_lstm = np.var(y_val_pred_lstm.flatten())
    stability_weight_xgb = 1 / (pred_var_xgb + 1e-6)
    stability_weight_lstm = 1 / (pred_var_lstm + 1e-6)
    total_weight_xgb = 0.4 * mse_weight_xgb + 0.3 * mae_weight_xgb + 0.2 * r2_weight_xgb + 0.1 * stability_weight_xgb
    total_weight_lstm = 0.4 * mse_weight_lstm + 0.3 * mae_weight_lstm + 0.2 * r2_weight_lstm + 0.1 * stability_weight_lstm
    total_weight = total_weight_xgb + total_weight_lstm
    alpha_xgb = total_weight_xgb / total_weight
    alpha_lstm = total_weight_lstm / total_weight
    return alpha_xgb, alpha_lstm

def simple_cold_item_forecast(cold_df):
    """
    對冷門商品進行簡單平均預測。

    參數：
        cold_df (pd.DataFrame): 冷門商品資料

    返回：
        cold_item_avg_dict (dict): 冷門商品的平均銷量字典
    """
    cold_item_avg = cold_df.groupby('ProductName')['Quantity'].mean().reset_index()
    cold_item_avg.columns = ['ProductName', 'AvgQty']
    cold_item_avg_dict = dict(zip(cold_item_avg['ProductName'], cold_item_avg['AvgQty']))
    return cold_item_avg_dict

def forecast_next_week_ensemble_dynamic(
    model_xgb, model_lstm, recent_main_df, cold_items_list, feature_cols,
    le_product, scaler=None, horizon=7, lookback=7
):
    """
    使用XGBoost和LSTM進行未來7天的動態權重融合預測，包含冷門商品處理。

    參數：
        model_xgb (MultiOutputRegressor): 訓練好的XGBoost模型
        model_lstm (Sequential): 訓練好的LSTM模型
        recent_main_df (pd.DataFrame): 最近的主力商品資料
        cold_items_list (list): 冷門商品名稱列表
        feature_cols (list): 特徵欄位名稱
        le_product (LabelEncoder): 商品名稱編碼器
        scaler (StandardScaler): 標準化器，預設為None
        horizon (int): 預測天數，預設為7
        lookback (int): 回顧天數，預設為7

    返回：
        forecast_df (pd.DataFrame): 預測結果
    """
    results = []
    grouped = recent_main_df.groupby('ProductName')
    
    for pname, subdf in grouped:
        if pname in cold_items_list:
            continue
        if len(subdf) < lookback:
            continue
        subdf = subdf.sort_values('SalesDate').iloc[-lookback:]
        X_input_3d = subdf[feature_cols].values.reshape(1, lookback, len(feature_cols))
        if scaler is not None:
            X_input_2d = X_input_3d.reshape(lookback, len(feature_cols))
            X_input_2d_scaled = scaler.transform(X_input_2d)
            X_input_3d = X_input_2d_scaled.reshape(1, lookback, len(feature_cols))
        X_input_flat = X_input_3d.reshape(1, -1)
        
        # 獲取 XGBoost 和 LSTM 的預測
        y_pred_xgb_log = model_xgb.predict(X_input_flat)
        y_pred_lstm_log = model_lstm.predict(X_input_3d, verbose=0)
        
        # 動態權重融合預測
        y_pred_xgb = np.expm1(y_pred_xgb_log)
        y_pred_lstm = np.expm1(y_pred_lstm_log)
        y_pred_ensemble = (alpha_xgb * y_pred_xgb + alpha_lstm * y_pred_lstm)
        
        for day_ahead in range(horizon):
            predict_date = subdf['SalesDate'].max() + pd.Timedelta(days=day_ahead + 1)
            results.append({
                'ProductName': pname,
                'ForecastDate': predict_date.date().isoformat(),
                'XGB_Forecast': float(y_pred_xgb[0, day_ahead]),
                'LSTM_Forecast': float(y_pred_lstm[0, day_ahead]),
                'Ensemble_Forecast': float(y_pred_ensemble[0, day_ahead])
            })
    
    # 冷門商品預測
    cold_item_avg_dict = simple_cold_item_forecast(cold_df)
    for citem in cold_items_list:
        forecast_avg = cold_item_avg_dict.get(citem, 2)
        for day_ahead in range(horizon):
            date_c = (recent_main_df['SalesDate'].max() + pd.Timedelta(days=day_ahead + 1)).date().isoformat()
            results.append({
                'ProductName': citem,
                'ForecastDate': date_c,
                'XGB_Forecast': None,
                'LSTM_Forecast': None,
                'Ensemble_Forecast': float(forecast_avg)
            })
    
    return pd.DataFrame(results)

def ensemble_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, main_df, cold_df, model_xgb, model_lstm, val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm, y_val_pred_xgb, y_val_pred_lstm):
    """
    執行動態權重融合、冷門商品預測和最終預測。

    參數：
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test (np.array): 訓練/驗證/測試資料
        main_df, cold_df (pd.DataFrame): 主力與冷門商品資料
        model_xgb (MultiOutputRegressor): 訓練好的XGBoost模型
        model_lstm (Sequential): 訓練好的LSTM模型
        val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm (float): 驗證集指標
        y_val_pred_xgb, y_val_pred_lstm (np.array): 驗證集預測結果

    返回：
        forecast_ens_df (pd.DataFrame): 最終預測結果
    """
    # 動態權重分配
    global alpha_xgb, alpha_lstm
    alpha_xgb, alpha_lstm = calculate_dynamic_weights(
        val_mse_xgb, val_mse_lstm, val_r2_xgb, val_r2_lstm, val_mae_xgb, val_mae_lstm, y_val_pred_xgb, y_val_pred_lstm
    )
    print(f"動態權重 -- XGB: {alpha_xgb:.3f}, LSTM: {alpha_lstm:.3f}")

    # 執行預測
    today = main_df['SalesDate'].max()
    recent_cut = today - pd.Timedelta(days=6)
    recent_main_df = main_df[main_df['SalesDate'] >= recent_cut]
    cold_items = cold_df['ProductName'].unique().tolist()

    forecast_ens_df = forecast_next_week_ensemble_dynamic(
        model_xgb=model_xgb,
        model_lstm=model_lstm,
        recent_main_df=recent_main_df,
        cold_items_list=cold_items,
        feature_cols=feature_cols,
        le_product=le_product,
        scaler=scaler,
        horizon=7,
        lookback=7
    )

    # 儲存模型
    joblib.dump(model_xgb, 'multi_store_xgb_final.pkl')
    model_lstm.save('multi_store_lstm_final.keras')

    # 儲存預測結果
    forecast_ens_df.to_csv('forecast_ensemble_final.csv', index=False)
    print("Ensemble 預測結果已輸出到 forecast_ensemble_final.csv")

    return forecast_ens_df

# 測試程式碼
if __name__ == "__main__":
    import data_preprocessing as dp
    import src.feature_engineering as fe
    import src.sequence_preparation as sp
    import src.model_training_xgb as xgb
    import src.model_training_lstm as lstm
    main_df, cold_df = dp.preprocess_data()
    main_df, feature_cols, le_product = fe.engineer_features(main_df)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = sp.split_and_scale_data(main_df, feature_cols)
    model_xgb, val_mse_xgb, val_r2_xgb, val_mae_xgb, y_val_pred_xgb = xgb.train_xgb_model(X_train_scaled, y_train, X_val_scaled, y_val)
    model_lstm, val_mse_lstm, val_r2_lstm, val_mae_lstm, y_val_pred_lstm, _ = lstm.train_lstm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    forecast_ens_df = ensemble_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, main_df, cold_df, model_xgb, model_lstm, val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm, y_val_pred_xgb, y_val_pred_lstm)
    print("預測完成")