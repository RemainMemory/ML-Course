import numpy as np


# 全连接层
class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 初始化全连接层
        self.num_input = num_input
        self.num_output = num_output
        self.init_param()  # 初始化参数

    def init_param(self, std=0.01):  # 初始化权重和偏置
        # 权重初始化为小随机数，偏置初始化为零
        self.weight = np.random.normal(0.0, std, (self.num_input, self.num_output))
        self.bias = np.zeros((1, self.num_output))
        # Adam参数初始化
        self.m = np.zeros_like(self.weight)  # 一阶矩变量
        self.v = np.zeros_like(self.weight)  # 二阶矩变量
        self.m_bias = np.zeros_like(self.bias)  # 偏置的一阶矩变量
        self.v_bias = np.zeros_like(self.bias)  # 偏置的二阶矩变量

    def forward(self, input):  # 前向传播计算
        self.input = input
        # 计算层的输出
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):  # 反向传播计算
        # 计算相对于权重的损失梯度
        self.d_weight = np.dot(self.input.T, top_diff)
        # 计算相对于偏置的损失梯度
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        # 计算传递到下一层的梯度
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):  # 使用梯度下降更新参数
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    def load_param(self, weight, bias):  # 加载参数
        self.weight = weight
        self.bias = bias

    def save_param(self):  # 保存参数
        return self.weight, self.bias


# ReLU激活层
class ReLULayer(object):
    def forward(self, input):  # ReLU前向传播计算
        self.input = input
        # 计算ReLU激活函数的输出
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, top_diff):  # ReLU反向传播计算
        # 梯度在输入大于0时为1，其余为0
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


# Softmax损失层
class SoftmaxLossLayer(object):
    def forward(self, input):  # Softmax前向传播计算
        input_max = np.max(input, axis=1, keepdims=True)  # 为了数值稳定性，减去最大值
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):  # 计算损失
        self.batch_size = self.prob.shape[0]
        # 创建one-hot编码的标签矩阵
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1
        # 计算交叉熵损失
        loss = -np.sum(self.label_onehot * np.log(self.prob)) / self.batch_size
        return loss

    def backward(self):  # Softmax反向传播计算
        # 计算相对于输入的损失梯度
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff


class DropoutLayer(object):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, input, is_train=True):
        if is_train:
            self.mask = (np.random.rand(*input.shape) < self.p) / self.p
            output = input * self.mask
        else:
            output = input
        return output

    def backward(self, top_diff):
        return top_diff * self.mask
