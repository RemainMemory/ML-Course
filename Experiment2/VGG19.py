import numpy as np
import struct
import os
from layers_1 import ReLULayer, FullyConnectedLayer, SoftmaxLossLayer


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = int((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        out_width = int((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # Apply padding to the input
        padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')

        # Initialize the output
        self.output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Perform the convolution operation
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                x_slice = padded_x[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.out_channels):
                    self.output[:, k, i, j] = np.sum(x_slice * self.weights[k, :, :, :], axis=(1, 2, 3)) + self.bias[k]

        return self.output


class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        # Initialize the output
        self.output = np.zeros((batch_size, channels, out_height, out_width))

        # Perform the max pooling operation
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, i, j] = np.max(x_slice, axis=(2, 3))

        return self.output


class SimpleVGG19:
    def __init__(self):
        # Define the architecture components
        # Convolutional blocks
        self.conv1_1 = ConvolutionalLayer(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = ConvolutionalLayer(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = MaxPoolingLayer(pool_size=2, stride=2)

        self.conv2_1 = ConvolutionalLayer(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = ConvolutionalLayer(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = MaxPoolingLayer(pool_size=2, stride=2)

        self.conv3_1 = ConvolutionalLayer(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = ConvolutionalLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = ConvolutionalLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = MaxPoolingLayer(pool_size=2, stride=2)

        # Fully connected layers
        self.fc1 = FullyConnectedLayer(num_input=256 * 3 * 3, num_output=4096)
        self.fc2 = FullyConnectedLayer(num_input=4096, num_output=4096)
        self.fc3 = FullyConnectedLayer(num_input=4096, num_output=10)  # 10 classes for MNIST

        # ReLU and Softmax layers
        self.relu = ReLULayer()
        self.softmax = SoftmaxLossLayer()

    def forward(self, x):
        # Forward pass through the convolutional blocks
        x = self.relu.forward(self.conv1_1.forward(x))
        x = self.relu.forward(self.conv1_2.forward(x))
        x = self.pool1.forward(x)

        x = self.relu.forward(self.conv2_1.forward(x))
        x = self.relu.forward(self.conv2_2.forward(x))
        x = self.pool2.forward(x)

        x = self.relu.forward(self.conv3_1.forward(x))
        x = self.relu.forward(self.conv3_2.forward(x))
        x = self.relu.forward(self.conv3_3.forward(x))
        x = self.pool3.forward(x)

        # Flatten the output for the fully connected layers
        x = x.reshape(x.shape[0], -1)

        # Forward pass through the fully connected layers
        x = self.relu.forward(self.fc1.forward(x))
        x = self.relu.forward(self.fc2.forward(x))
        x = self.fc3.forward(x)  # No ReLU activation before the softmax layer

        # Softmax layer
        prob = self.softmax.forward(x)
        return prob


class MNISTLoader:
    def load_mnist(self, path, is_images=True):
        # Open the binary file
        with open(path, 'rb') as file:
            bin_data = file.read()

        # Read the header information from the binary file
        if is_images:  # If reading image data
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
            data_size = num_images * num_rows * num_cols
            mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
            mat_data = np.reshape(mat_data, newshape=(num_images, num_rows, num_cols))
        else:  # If reading label data
            fmt_header = '>ii'
            magic, num_items = struct.unpack_from(fmt_header, bin_data, 0)
            data_size = num_items
            mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
            mat_data = np.reshape(mat_data, newshape=(num_items,))

        return mat_data


# Instantiate the model
model = SimpleVGG19()

# Define the paths for the MNIST test dataset
mnist_dir = './Experiment2'
test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

# Load and preprocess the test dataset
mnist_loader = MNISTLoader()
test_images = mnist_loader.load_mnist(test_images_path, is_images=True)
test_labels = mnist_loader.load_mnist(test_labels_path, is_images=False)
test_images = test_images.reshape(-1, 1, 28, 28) / 255.0  # Normalize and add channel dimension
test_labels = test_labels.flatten()

# Instantiate the model
model = SimpleVGG19()

# Define the paths for the MNIST test dataset
mnist_dir = '..'
test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

# Load and preprocess the test dataset
mnist_loader = MNISTLoader()
test_images = mnist_loader.load_mnist(test_images_path, is_images=True)
test_labels = mnist_loader.load_mnist(test_labels_path, is_images=False)
test_images = test_images.reshape(-1, 1, 28, 28) / 255.0  # Normalize and add channel dimension
test_labels = test_labels.flatten()


# Function to evaluate the model
def evaluate_model(model, test_images, test_labels, batch_size=256):
    num_batches = len(test_images) // batch_size
    correct_predictions = 0

    for i in range(num_batches):
        batch_images = test_images[i * batch_size:(i + 1) * batch_size]
        batch_labels = test_labels[i * batch_size:(i + 1) * batch_size]

        # Forward pass
        predictions = model.forward(batch_images)
        predicted_classes = np.argmax(predictions, axis=1)

        # Accumulate correct predictions
        correct_predictions += np.sum(predicted_classes == batch_labels)

    # Calculate accuracy
    accuracy = correct_predictions / len(test_labels)
    return accuracy


# Evaluate the model
model_accuracy = evaluate_model(model, test_images, test_labels)
print(f"Model accuracy: {model_accuracy * 100:.2f}%")
