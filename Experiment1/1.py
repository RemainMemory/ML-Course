import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm

# 生成模拟记忆任务数据
np.random.seed(0)
x_data = np.linspace(1, 10, 100)
y_data_power = 1 / (x_data ** 2) + np.random.normal(0, 0.05, x_data.size)
y_data_exponential = np.exp(-x_data / 3) + np.random.normal(0, 0.05, x_data.size)


# 定义冥函数和指数函数
def power_function(x, a, b):
    return a / (x ** b)


def exponential_function(x, a, b):
    return a * np.exp(-b * x)


# 使用最小二乘法进行参数估计
params_power, _ = curve_fit(power_function, x_data, y_data_power)
params_exponential, _ = curve_fit(exponential_function, x_data, y_data_exponential)
print("Least Squares - Power Model Parameters:", params_power)
print("Least Squares - Exponential Model Parameters:", params_exponential)


# 定义最大似然估计函数
def mle_power(params):
    a, b = params
    pred = power_function(x_data, a, b)
    return -np.sum(norm.logpdf(y_data_power, pred, 0.05))


def mle_exponential(params):
    a, b = params
    pred = exponential_function(x_data, a, b)
    return -np.sum(norm.logpdf(y_data_exponential, pred, 0.05))


# 使用最大似然法进行参数估计
mle_params_power = minimize(mle_power, params_power).x
mle_params_exponential = minimize(mle_exponential, params_exponential).x
print("Maximum Likelihood - Power Model Parameters:", mle_params_power)
print("Maximum Likelihood - Exponential Model Parameters:", mle_params_exponential)


# 计算模型拟合度
def compute_rss(y, y_pred):
    return np.sum((y - y_pred) ** 2)


# 使用估计的参数计算拟合度
y_pred_power = power_function(x_data, *mle_params_power)
y_pred_exponential = exponential_function(x_data, *mle_params_exponential)

rss_power = compute_rss(y_data_power, y_pred_power)
rss_exponential = compute_rss(y_data_exponential, y_pred_exponential)
print("RSS - Power Model:", rss_power)
print("RSS - Exponential Model:", rss_exponential)

# 计算AIC和BIC
n = len(x_data)
k_power = len(mle_params_power)
k_exponential = len(mle_params_exponential)

aic_power = n * np.log(rss_power / n) + 2 * k_power
bic_power = n * np.log(rss_power / n) + np.log(n) * k_power
aic_exponential = n * np.log(rss_exponential / n) + 2 * k_exponential
bic_exponential = n * np.log(rss_exponential / n) + np.log(n) * k_exponential
print("AIC - Power Model:", aic_power)
print("BIC - Power Model:", bic_power)
print("AIC - Exponential Model:", aic_exponential)
print("BIC - Exponential Model:", bic_exponential)

# 绘制原始数据和模型拟合结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x_data, y_data_power, label='Original Data')
plt.plot(x_data, y_pred_power, color='red', label='Fitted Power Model')
plt.title('Power Function Model')
plt.xlabel('Time')
plt.ylabel('Memory Retention')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_data, y_data_exponential, label='Original Data')
plt.plot(x_data, y_pred_exponential, color='red', label='Fitted Exponential Model')
plt.title('Exponential Function Model')
plt.xlabel('Time')
plt.ylabel('Memory Retention')
plt.legend()

plt.tight_layout()
plt.show()
