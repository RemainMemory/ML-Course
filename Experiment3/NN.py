# 导入库和模块
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

# 数据预处理
# 读取CSV文件
raw_data = pd.read_csv("./Netflix Userbase.csv")
# 显示数据信息
raw_data.info()
# 显示数据描述
raw_data.describe()
# 删除不必要的列
raw_data.drop(['User ID'], axis=1)
# 转换日期格式并计算订阅时长
raw_data['Join Date'] = pd.to_datetime(raw_data['Join Date'])
raw_data['Last Payment Date'] = pd.to_datetime(raw_data['Last Payment Date'])
raw_data['Subscribed Since'] = (raw_data['Last Payment Date'] - raw_data['Join Date']).dt.days

# 数据可视化
# 画出不同订阅类型的月收入柱状图
sm = raw_data.groupby('Subscription Type').sum(numeric_only=True).reset_index()
sns.barplot(data=sm, x='Subscription Type', y='Monthly Revenue')
plt.show()
# 画出不同年龄组的订阅数量柱状图
sns.countplot(x='Age', data=raw_data)
plt.title("Subscription by Age Groups")
plt.show()
# 画出相关性热图
cols = raw_data[['Monthly Revenue', 'Age', 'Subscribed Since']].corr()
sns.heatmap(cols, annot=True, cmap='coolwarm')
plt.title("Hot map")
plt.show()

# 数据编码和标准化
# 删除不必要的列并进行One-Hot编码
raw_data = raw_data.drop(columns=['Join Date', 'Last Payment Date', 'User ID', 'Plan Duration'], axis=1)
data_labels = raw_data["Subscribed Since"]
data_encoded = pd.get_dummies(raw_data.drop(["Subscribed Since"], axis=1))
# 标准化数据
StandardScaler = StandardScaler()
StandardScaler.fit(data_encoded)
scaled_inputs = StandardScaler.transform(data_encoded)
# 数据集切分
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, data_labels, test_size=0.2, random_state=40)

# 模型构建和训练
# 构建神经网络模型
tf.random.set_seed(45)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, 'relu', input_shape=(21,)))
model.add(tf.keras.layers.Dense(4, 'relu'))
model.add(tf.keras.layers.Dense(1, None))
# 编译模型
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics="mae")
# 训练模型
history = model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=500, verbose=2, validation_split=0.2)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training_and_Validation_Loss.png')
plt.show()

# 模型评估
# 使用测试集进行预测
predictions = model.predict(x_test)
# 计算MAE
mae = mean_absolute_error(y_test, predictions)
print(mae)



# 训练集和测试集的拟合图
train_predictions = model.predict(x_train)
plt.figure()
plt.scatter(y_train, train_predictions)
plt.xlabel('True Values [Subscribed Since]')
plt.ylabel('Predictions [Subscribed Since]')
plt.title('Training Data: True vs Predicted')
plt.savefig('Training_True_vs_Predicted.png')
plt.show()

# 预测图
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel('True Values [Subscribed Since]')
plt.ylabel('Predictions [Subscribed Since]')
plt.title('Test Data: True vs Predicted')
plt.savefig('Test_True_vs_Predicted.png')
plt.show()

# 打印MAE
print(f"Mean Absolute Error: {mae}")