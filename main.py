import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 建立 LSTM 模型
def build_multivariate_lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(None, 4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

# 创建数据集
def create_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:(i + time_steps)].values)
        Y.append(data.iloc[i + time_steps]['Close'])
    return np.array(X), np.array(Y)

# 股票代码
stock_codes = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011',
               '000012', '000014', '000016', '000017', '600004']

time_steps = 10

for code in stock_codes:
    df = pd.read_csv(code + '.csv', encoding='GBK')
    df = df.dropna()

    features = df[['Open', 'High', 'Low', 'Close']]
    means = features.mean()
    stds = features.std()
    features = (features - means) / stds

    train_size = int(len(features) * 0.8)
    train_data = features[:train_size]
    test_data = features[train_size:]

    X_train, Y_train = create_dataset(train_data, time_steps)
    X_test, Y_test = create_dataset(test_data, time_steps)

    model = build_multivariate_lstm_model()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2, callbacks=[early_stopping_callback])

    test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
    print('股票代码 {} 的测试损失 loss: {:.4f}, 测试 MAE: {:.4f}'.format(code, test_loss, test_mae))

    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(Y_test, model.predict(X_test)))
    print('均方根误差: {:.4f}'.format(rmse))

    # 计算 R^2
    r2 = r2_score(Y_test, model.predict(X_test))
    print('R^2 分数: {:.4f}'.format(r2))

    # 创建并保存预测分数图表
    # 创建并保存预测分数图表
    if not os.path.exists("prediction_scores"):
        os.makedirs("prediction_scores")
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(['Test Loss', 'Test MAE', 'RMSE', 'R^2'], [test_loss, test_mae, rmse, r2])
    plt.title('Prediction Scores for Stock Code ' + str(code))
    plt.ylabel('Score')

    # 在每一个条形上方添加数值
    for rect in bar_plot:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

    plt.savefig('prediction_scores/' + code + '_prediction_scores.png')
    plt.close()


    future_data = []
    last_data = train_data[-time_steps:][['Open', 'High', 'Low', 'Close']].values.reshape(-1, time_steps, 4)
    for i in range(7):
        predict = model.predict(last_data)
        future_data.append(predict[0][0])

        new_data = np.zeros((1, time_steps, 4))
        new_data[:, :-1, :] = last_data[:, 1:, :]
        new_data[:, -1, :3] = last_data[:, -1, 1:4]
        new_data[:, -1, 3] = predict
        last_data = new_data

    future_data = np.array(future_data) * stds['Close'] + means['Close']
    future_date_range = pd.date_range(start=df['Date'].iloc[-1], periods=7, freq='D')
    future_df = pd.DataFrame({'date': future_date_range, 'future_data': future_data})

    if not os.path.exists("future_datas"):
        os.makedirs("future_datas")
    future_df.to_csv("future_datas/" + code + '_future_data.csv', index=False)

    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/" + code + "_model.h5")

    # plot actual vs predicted values
    if not os.path.exists("prediction_plots"):
        os.makedirs("prediction_plots")
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, label='Actual')
    plt.plot(model.predict(X_test), label='Predicted')
    plt.title('Actual vs Predicted Close Prices for Stock Code ' + str(code))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('prediction_plots/' + code + '_pred_vs_actual.png')
    plt.close()