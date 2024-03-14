import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(6, input_shape=(1, 2)),  # 1 time step, 2 features
    tf.keras.layers.Dense(2)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

# 生成训练数据
num_samples = 1000
X_train = np.random.rand(num_samples, 1, 2) * 20  # 随机水位变化
X_train[:, :, 1] = 80  # 设置初始水位为80
Y_train = np.array([[100, 100] for _ in range(num_samples)])  # 理想水位

# 训练模型
history = model.fit(X_train, Y_train, epochs=50, batch_size=10)

# 绘制训练损失
plt.plot(history.history['loss'])
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 测试模型
X_test = np.array([[[np.random.rand() * 20, 80]]])  # 新的水位变化和初始水位
Y_pred = model.predict(X_test)

# 绘制测试结果
plt.scatter([1], X_test[0, 0, 0], color='blue', label='Actual Water Level Change')
plt.scatter([1], Y_pred[0, 0], color='red', label='Predicted Water Level Change')
plt.axhline(y=100, color='green', linestyle='-', label='Desired Water Level')
plt.legend()
plt.title('Test Result')
plt.ylabel('Water Level')
plt.xticks([])
plt.show()

print("测试输入 (水位变化, 初始水位):", X_test[0][0])
print("预测水位变化:", Y_pred[0])
