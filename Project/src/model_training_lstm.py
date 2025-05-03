# model_training_lstm.py
# 負責LSTM模型的訓練與評估，專注於時間序列預測。

# 導入必要的套件
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def custom_mse(y_true, y_pred):
    """
    自定義損失函數，對高銷量數據給予更高懲罰。

    參數：
        y_true (tf.Tensor): 真實值
        y_pred (tf.Tensor): 預測值

    返回：
        loss (tf.Tensor): 加權均方誤差
    """
    error = y_true - y_pred
    mean_y = tf.reduce_mean(y_true)
    weights = tf.where(y_true > mean_y * 1.5, 3.0, tf.where(y_true > mean_y, 2.0, 1.0))
    return tf.reduce_mean(weights * tf.square(error))

def build_lstm_model(input_shape, horizon=7, lstm_units=96, dropout_rate=0.2):
    """
    建構雙向LSTM模型。

    參數：
        input_shape (tuple): 輸入形狀 (lookback, n_features)
        horizon (int): 預測天數，預設為7
        lstm_units (int): LSTM單元數，預設為96
        dropout_rate (float): Dropout比例，預設為0.2

    返回：
        model (Sequential): LSTM模型
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_units, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units // 2, activation='relu')))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)))
    model.add(Dropout(0.2))
    model.add(Dense(horizon))
    model.compile(optimizer=Adam(3e-4), loss=custom_mse)
    return model

def train_lstm_model(X_train_scaled, y_train, X_val_scaled, y_val):
    """
    訓練LSTM模型，使用早停機制避免過擬合。

    參數：
        X_train_scaled (np.array): 訓練集特徵（標準化後）
        y_train (np.array): 訓練集目標
        X_val_scaled (np.array): 驗證集特徵（標準化後）
        y_val (np.array): 驗證集目標

    返回：
        model_lstm (Sequential): 訓練好的LSTM模型
        val_mse_lstm, val_r2_lstm, val_mae_lstm (float): 驗證集評估指標
        y_val_pred_lstm (np.array): 驗證集預測結果
        history (dict): 訓練歷史記錄
    """
    # 定義輸入形狀
    lookback = X_train_scaled.shape[1]
    n_features = X_train_scaled.shape[2]
    input_shape = (lookback, n_features)

    # 建構LSTM模型
    model_lstm = build_lstm_model(input_shape, horizon=7, lstm_units=96, dropout_rate=0.2)

    # 加入早停機制
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 訓練模型
    history = model_lstm.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # 預測並反變換
    y_val_pred_lstm_log = model_lstm.predict(X_val_scaled)
    y_val_pred_lstm = np.expm1(y_val_pred_lstm_log)
    y_val_original = np.expm1(y_val)

    # 計算驗證集評估指標
    val_mse_lstm = mean_squared_error(y_val_original.flatten(), y_val_pred_lstm.flatten())
    val_r2_lstm = r2_score(y_val_original.flatten(), y_val_pred_lstm.flatten())
    val_mae_lstm = mean_absolute_error(y_val_original.flatten(), y_val_pred_lstm.flatten())
    print("LSTM Val MSE:", val_mse_lstm, "LSTM Val R2:", val_r2_lstm, "LSTM Val MAE:", val_mae_lstm)

    return model_lstm, val_mse_lstm, val_r2_lstm, val_mae_lstm, y_val_pred_lstm, history

# 測試程式碼
if __name__ == "__main__":
    import data_preprocessing as dp
    import feature_engineering as fe
    import sequence_preparation as sp
    main_df, _ = dp.preprocess_data()
    main_df, feature_cols, _ = fe.engineer_features(main_df)
    X_train_scaled, y_train, X_val_scaled, y_val, _, _, _ = sp.split_and_scale_data(main_df, feature_cols)
    model_lstm, mse, r2, mae, _, _ = train_lstm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    print("LSTM訓練完成")