import pandas as pd
import numpy as np
import tensorflow as tf
import os

def build_multivariate_lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(None, 4), return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))
    return model

stock_codes = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000014', '000016', '000017', '600004']

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

    model = build_multivariate_lstm_model()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
    model.fit(train_data.values.reshape(-1, 1, 4), train_data['Close'].values.reshape(-1, 1), epochs=100, batch_size=32, verbose=2)

    test_loss, test_mae = model.evaluate(test_data.values.reshape(-1, 1, 4), test_data['Close'].values.reshape(-1, 1), verbose=0)
    print('股票代码 {} 的测试 MAE: {:.4f}, 测试损失: {:.4f}'.format(code, test_mae, test_loss))

    future_data = []
    last_data = train_data[-1:][['Open', 'High', 'Low', 'Close']].values.reshape(-1, 1, 4)
    for i in range(30):
        predict = model.predict(last_data)
        future_data.append(predict[0][0])

        new_data = np.zeros((1, 1, 4))
        new_data[:, :, :3] = last_data[:, :, 1:4]
        new_data[:, :, 3] = predict
        last_data = new_data

    future_data = np.array(future_data) * stds['Close'] + means['Close']
    future_date_range = pd.date_range(start=df['Date'].iloc[-1], periods=30, freq='D')
    future_df = pd.DataFrame({'date': future_date_range, 'future_data': future_data})

    if not os.path.exists("future_datas"):
        os.makedirs("future_datas")
    future_df.to_csv("future_datas/" + code + '_future_data.csv', index=False)

    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/" + code + "_model.h5")
