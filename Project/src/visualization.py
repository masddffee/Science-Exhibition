# visualization.py
# 負責生成科展展示用的圖表，展示模型改進和預測效果。

# 導入必要的套件
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(main_df, y_val, y_val_pred_xgb, y_val_pred_lstm, y_test, y_test_pred_stacking, history, val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm, test_mse_stacking, test_r2_stacking, test_mae_stacking, alpha_xgb, alpha_lstm):
    """
    生成科展展示用的圖表，包括評估指標、動態權重、殘差圖等。

    參數：
        main_df (pd.DataFrame): 主力商品資料
        y_val, y_val_pred_xgb, y_val_pred_lstm, y_test, y_test_pred_stacking (np.array): 驗證與測試預測結果
        history (dict): LSTM訓練歷史記錄
        val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm (float): 驗證集指標
        test_mse_stacking, test_r2_stacking, test_mae_stacking (float): 測試集指標
        alpha_xgb, alpha_lstm (float): 動態權重
    """
    # 儲存未優化前的評估指標
    before_metrics = {
        'XGBoost': {'MSE': 228.32, 'R2': 0.5714, 'MAE': 5.186},
        'LSTM': {'MSE': 295.03, 'R2': 0.4462, 'MAE': 6.080}
    }

    # 圖表 1：評估指標對照條狀圖（改進前後對比）
    metrics = ['MSE', 'R2', 'MAE']
    xgb_metrics_before = [before_metrics['XGBoost']['MSE'], before_metrics['XGBoost']['R2'], before_metrics['XGBoost']['MAE']]
    lstm_metrics_before = [before_metrics['LSTM']['MSE'], before_metrics['LSTM']['R2'], before_metrics['LSTM']['MAE']]
    xgb_metrics_after = [val_mse_xgb, val_r2_xgb, val_mae_xgb]
    lstm_metrics_after = [val_mse_lstm, val_r2_lstm, val_mae_lstm]
    stacking_metrics = [test_mse_stacking, test_r2_stacking, test_mae_stacking]

    x = np.arange(len(metrics))
    width = 0.15

    plt.figure(figsize=(12, 6))
    plt.bar(x - width*2, xgb_metrics_before, width, label='XGBoost (Before)', color='lightblue')
    plt.bar(x - width, xgb_metrics_after, width, label='XGBoost (After)', color='blue')
    plt.bar(x, lstm_metrics_before, width, label='LSTM (Before)', color='lightcoral')
    plt.bar(x + width, lstm_metrics_after, width, label='LSTM (After)', color='orange')
    plt.bar(x + width*2, stacking_metrics, width, label='Stacking (Final)', color='green')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics Comparison (Before vs. After Optimization)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluation_metrics_comparison_final.png')

    # 圖表 2：動態權重計算示意圖
    mse_weight_xgb = 0.4 * alpha_xgb
    mse_weight_lstm = 0.4 * alpha_lstm
    mae_weight_xgb = 0.3 * alpha_xgb
    mae_weight_lstm = 0.3 * alpha_lstm
    r2_weight_xgb = 0.2 * alpha_xgb
    r2_weight_lstm = 0.2 * alpha_lstm
    stability_weight_xgb = 0.1 * alpha_xgb
    stability_weight_lstm = 0.1 * alpha_lstm

    plt.figure(figsize=(8, 6))
    plt.bar(['XGBoost', 'LSTM'], [mse_weight_xgb, mse_weight_lstm], label='MSE Weight (40%)', color='blue')
    plt.bar(['XGBoost', 'LSTM'], [mae_weight_xgb, mae_weight_lstm], bottom=[mse_weight_xgb, mse_weight_lstm], label='MAE Weight (30%)', color='orange')
    plt.bar(['XGBoost', 'LSTM'], [r2_weight_xgb, r2_weight_lstm], bottom=[mse_weight_xgb+mae_weight_xgb, mse_weight_lstm+mae_weight_lstm], label='R2 Weight (20%)', color='green')
    plt.bar(['XGBoost', 'LSTM'], [stability_weight_xgb, stability_weight_lstm], bottom=[mse_weight_xgb+mae_weight_xgb+r2_weight_xgb, mse_weight_lstm+mae_weight_lstm+r2_weight_lstm], label='Stability Weight (10%)', color='red')
    plt.xlabel('Model')
    plt.ylabel('Weight Contribution')
    plt.title('Dynamic Weight Calculation Breakdown')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dynamic_weight_calculation_final.png')

    # 圖表 3：殘差圖
    residuals_xgb = np.expm1(y_val).flatten() - y_val_pred_xgb.flatten()
    residuals_lstm = np.expm1(y_val).flatten() - y_val_pred_lstm.flatten()
    residuals_stacking = np.expm1(y_test).flatten() - y_test_pred_stacking.flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_val_pred_xgb.flatten(), residuals_xgb, label='XGBoost', color='blue', alpha=0.5)
    plt.scatter(y_val_pred_lstm.flatten(), residuals_lstm, label='LSTM', color='orange', alpha=0.5)
    plt.scatter(y_test_pred_stacking.flatten(), residuals_stacking, label='Stacking', color='green', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot for XGBoost, LSTM, and Stacking (Final Optimization)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('residual_plot_final.png')

    # 圖表 4：LSTM 學習曲線
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('LSTM Training and Validation Loss Over Epochs (Final Optimization)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lstm_learning_curve_final.png')

    # 圖表 5：誤差分佈直方圖
    errors_xgb = np.expm1(y_val).flatten() - y_val_pred_xgb.flatten()
    errors_lstm = np.expm1(y_val).flatten() - y_val_pred_lstm.flatten()
    errors_stacking = np.expm1(y_test).flatten() - y_test_pred_stacking.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(errors_xgb, bins=50, label='XGBoost', color='blue', alpha=0.5)
    plt.hist(errors_lstm, bins=50, label='LSTM', color='orange', alpha=0.5)
    plt.hist(errors_stacking, bins=50, label='Stacking', color='green', alpha=0.5)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution for XGBoost, LSTM, and Stacking (Final Optimization)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('error_distribution_histogram_final.png')

    # 圖表 6：數據分佈圖（展示對數變換效果）
    plt.figure(figsize=(10, 6))
    plt.hist(main_df['Quantity'], bins=50, label='Original Quantity', color='blue', alpha=0.5)
    plt.hist(main_df['Log_Quantity'], bins=50, label='Log-Transformed Quantity', color='orange', alpha=0.5)
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Quantity Before and After Log Transformation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('quantity_distribution_final.png')

    # 圖表 7：改進過程總結圖（MSE 變化）
    stages = ['Before Optimization', 'After First Optimization', 'After Final Optimization']
    mse_values_xgb = [before_metrics['XGBoost']['MSE'], val_mse_xgb, val_mse_xgb]
    mse_values_lstm = [before_metrics['LSTM']['MSE'], val_mse_lstm, val_mse_lstm]
    mse_values_stacking = [None, None, test_mse_stacking]

    plt.figure(figsize=(10, 6))
    plt.plot(stages, mse_values_xgb, marker='o', label='XGBoost MSE', color='blue')
    plt.plot(stages, mse_values_lstm, marker='o', label='LSTM MSE', color='orange')
    plt.plot(stages[-1], mse_values_stacking[-1], marker='o', label='Stacking MSE', color='green')
    plt.xlabel('Optimization Stage')
    plt.ylabel('MSE')
    plt.title('MSE Improvement Across Optimization Stages')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mse_improvement_stages.png')

# 測試程式碼
if __name__ == "__main__":
    import data_preprocessing as dp
    import 完整區分程式.src.feature_engineering as fe
    import 完整區分程式.src.sequence_preparation as sp
    import 完整區分程式.src.model_training_xgb as xgb
    import 完整區分程式.src.model_training_lstm as lstm
    import 完整區分程式.src.model_ensemble as ens
    main_df, cold_df = dp.preprocess_data()
    main_df, feature_cols, le_product = fe.engineer_features(main_df)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler = sp.split_and_scale_data(main_df, feature_cols)
    model_xgb, val_mse_xgb, val_r2_xgb, val_mae_xgb, y_val_pred_xgb = xgb.train_xgb_model(X_train_scaled, y_train, X_val_scaled, y_val)
    model_lstm, val_mse_lstm, val_r2_lstm, val_mae_lstm, y_val_pred_lstm, history = lstm.train_lstm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    forecast_ens_df = ens.ensemble_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, main_df, cold_df, model_xgb, model_lstm)
    generate_visualizations(main_df, y_val, y_val_pred_xgb, y_val_pred_lstm, y_test, y_test_pred_stacking, history, val_mse_xgb, val_r2_xgb, val_mae_xgb, val_mse_lstm, val_r2_lstm, val_mae_lstm, test_mse_stacking, test_r2_stacking, test_mae_stacking, alpha_xgb, alpha_lstm)
    print("圖表生成完成")