# main.py
# 主執行檔案，調用各模塊執行完整流程。

# 導入必要的系統模組
import sys
import os

# 動態添加 src 目錄到模組搜尋路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 導入各模塊
import data_preprocessing as dp
import feature_engineering as fe
import sequence_preparation as sp
import model_training_xgb as xgb
import model_training_lstm as lstm
import model_ensemble as ens
import visualization as vis

# 執行完整流程
def main():
    # 1. 資料前處理
    main_df, cold_df = dp.preprocess_data()
    
    # 2. 特徵工程
    main_df, feature_cols, le_product = fe.engineer_features(main_df)
    
    # 3. 序列資料準備
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = sp.split_and_scale_data(main_df, feature_cols)
    
    # 4. 訓練XGBoost模型
    model_xgb, val_mse_xgb, val_r2_xgb, val_mae_xgb, y_val_pred_xgb = xgb.train_xgb_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 5. 訓練LSTM模型
    model_lstm, val_mse_lstm, val_r2_lstm, val_mae_lstm, y_val_pred_lstm, history = lstm.train_lstm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 6. 模型集成與預測
    forecast_ens_df = ens.ensemble_model(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
        main_df, cold_df, model_xgb, model_lstm,
        val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm,
        y_val_pred_xgb, y_val_pred_lstm
    )
    
    # 7. 生成圖表
    # 從 model_ensemble.py 中獲取 alpha_xgb 和 alpha_lstm
    alpha_xgb = ens.alpha_xgb
    alpha_lstm = ens.alpha_lstm
    
    vis.generate_visualizations(
        main_df, y_val, y_val_pred_xgb, y_val_pred_lstm, y_test,
        history, val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm,
        alpha_xgb, alpha_lstm
    )
    print("流程執行完成！")

if __name__ == "__main__":
    main()