import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

@tf.function
def hypothesis(W):
    return tf.multiply(W, X)

@tf.function
def cost(W):
    return tf.reduce_mean(tf.square(hypothesis(W) - Y))

W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_W = tf.Variable(feed_W, dtype=tf.float32)

    curr_cost = cost(feed_W)

    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
