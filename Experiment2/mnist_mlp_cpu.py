import os
import struct
import numpy as np

from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

# For example:
MNIST_DIR = './'
TRAIN_IMAGES = 'train-images-idx3-ubyte'
TRAIN_LABELS = 'train-labels-idx1-ubyte'
TEST_IMAGES = 't10k-images-idx3-ubyte'
TEST_LABELS = 't10k-labels-idx1-ubyte'


class MNISTLoader:
    def load_mnist(self, file_dir, is_images=True):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        if is_images:  # 读取图像数据
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:  # 读取标记数据
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1

        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])

        return mat_data

    def load_data(self):
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_IMAGES), is_images=True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABELS), is_images=False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_IMAGES), is_images=True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABELS), is_images=False)

        self.train_data = np.append(train_images, train_labels.reshape(-1, 1), axis=1)
        self.test_data = np.append(test_images, test_labels.reshape(-1, 1), axis=1)


# MNIST多层感知机(MLP)网络
class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=2,
                 print_iter=100):
        # 神经网络初始化
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.build_model()  # 建立模型

    def build_model(self):
        # 建立三层神经网络结构
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()  # 第二个ReLU激活层
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()

        # 需要更新参数的层列表
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        # 初始化神经网络参数
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):
        # 神经网络的前向传播
        h1 = self.fc1.forward(input)
        h1_relu = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1_relu)  # 第二层全连接的前向传播
        h2_relu = self.relu2.forward(h2)  # 第二层ReLU激活函数的前向传播
        h3 = self.fc3.forward(h2_relu)  # 第三层全连接的前向传播
        prob = self.softmax.forward(h3)  # Softmax层的前向传播
        return prob

    def backward(self):
        # 神经网络的反向传播
        dloss = self.softmax.backward()  # 反向传播开始于损失层
        dh3 = self.fc3.backward(dloss)  # 第三层全连接的反向传播
        dh2_relu = self.relu2.backward(dh3)  # 第二层ReLU的反向传播
        dh2 = self.fc2.backward(dh2_relu)  # 第二层全连接的反向传播
        dh1_relu = self.relu1.backward(dh2)  # 第一层ReLU的反向传播
        dh1 = self.fc1.backward(dh1_relu)  # 第一层全连接的反向传播
        return dh1

    def update(self, lr):
        # 神经网络参数更新
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        # 保存神经网络参数
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def shuffle_data(self):
        # 随机打乱数据
        indices = np.arange(self.train_data.shape[0])
        np.random.shuffle(indices)
        self.train_data = self.train_data[indices]

    def load_data(self):
        loader = MNISTLoader()
        self.train_data = loader.load_mnist(os.path.join(MNIST_DIR, TRAIN_IMAGES), is_images=True)
        self.train_labels = loader.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABELS), is_images=False)
        self.test_data = loader.load_mnist(os.path.join(MNIST_DIR, TEST_IMAGES), is_images=True)
        self.test_labels = loader.load_mnist(os.path.join(MNIST_DIR, TEST_LABELS), is_images=False)
        # Combine the images and labels into a single array for training and testing
        self.train_data = np.hstack((self.train_data, self.train_labels.reshape(-1, 1)))
        self.test_data = np.hstack((self.test_data, self.test_labels.reshape(-1, 1)))

    def train(self, train_data):
        # 训练函数主体
        self.train_data = train_data
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()  # 打乱数据
            for idx_batch in range(max_batch):
                batch_slice = slice(idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size)
                batch_images = self.train_data[batch_slice, :-1]  # 获取图像数据
                batch_labels = self.train_data[batch_slice, -1]  # 获取标签数据
                prob = self.forward(batch_images)  # 前向传播
                loss = self.softmax.get_loss(batch_labels)  # 计算损失
                self.backward()  # 反向传播
                self.update(self.lr)  # 更新参数
                if idx_batch % self.print_iter == 0:
                    print(f'Epoch {idx_epoch}, Iter {idx_batch}, Loss: {loss:.6f}')

    def load_model(self, param_dir):
        # 加载神经网络参数
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    def evaluate(self, test_data):
        # 推断函数主体
        self.test_data = test_data
        correct_predictions = 0
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            correct_predictions += np.sum(
                pred_labels == self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, -1])

        accuracy = correct_predictions / self.test_data.shape[0]
        print('Accuracy in test set: %f' % accuracy)


def build_mnist_mlp(param_dir='weight.npy'):
    # 构建、训练和保存MNIST MLP模型
    h1, h2, e = 256, 128, 20
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()  # 加载数据
    mlp.build_model()
    mlp.init_model()
    mlp.train(mlp.train_data)  # 传入训练数据
    mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    return mlp


if __name__ == '__main__':
    # 主程序入口，构建、训练并评估MLP模型
    mlp = build_mnist_mlp()
    mlp.evaluate(mlp.test_data)  # 传入测试数据
