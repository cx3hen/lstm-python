import pandas as pd
import numpy as np
import tensorflow as tf
import os


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


def create_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:(i + time_steps)].values)
        Y.append(data.iloc[i + time_steps]['Close'])
    return np.array(X), np.array(Y)


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

import matplotlib.pyplot as plt


# 生成图表的函数
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_error_distribution(y_true, y_pred, title):
    error = y_pred - y_true
    plt.figure(figsize=(14, 7))
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


# 对于每个股票代码，生成预测值并绘制图表
for code in stock_codes:
    model = tf.keras.models.load_model("models/" + code + "_model.h5")
    X_test, Y_test = create_dataset(test_data, time_steps)
    Y_pred = model.predict(X_test).flatten()

    plot_results(Y_test, Y_pred, 'Stock Code: ' + code + ' Actual vs Predicted')
    plot_error_distribution(Y_test, Y_pred, 'Stock Code: ' + code + ' Error Distribution')
