import tensorflow as tf
from tensorflow.keras.layers import Dense

class NeuralNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.dense1 = Dense(hidden_size, activation='relu', input_shape=(input_size,))
        self.dense2 = Dense(hidden_size, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
