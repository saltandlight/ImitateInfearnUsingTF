import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

y_data = np.array([[0],
                   [1],
                   [1], [0]], dtype=np.float32)

# Neural Network 적용
W1 = tf.Variable(tf.random.normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random.normal([10]), name='bias1')

@tf.function
def layer(X):
    return tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random.normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random.normal([10]), name='bias2')

@tf.function
def hypothesis(X):
    layer1 = layer(X)
    return tf.sigmoid(tf.matmul(layer1, W2) + b2)

@tf.function
def cost(X, Y):
    hypo = hypothesis(X)
    return -tf.reduce_mean(Y * tf.math.log(hypo) + (1-Y) * tf.math.log(1 - hypo))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

@tf.function
def predict(X):
    hypo = hypothesis(X)
    return tf.cast(hypo > 0.5, dtype=tf.float32)

@tf.function
def accuracy(X, Y):
    pred = predict(X)
    return tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

X_val = tf.Variable(x_data, dtype=tf.float32)
Y_val = tf.Variable(y_data, dtype=tf.float32)

for step in range(10001):
    with tf.GradientTape() as tp:
        cost_val = cost(X_val, Y_val)

    train = optimizer.minimize(cost_val, var_list=[W1, W2, b1, b2], tape=tp)
    if step % 1000 == 0:
        print(f'step = {step:06}, cost : {cost_val:.10}, '
              f'weight= {tf.squeeze(W1.numpy())}, '
              f'{tf.squeeze(W2.numpy())}')

h = hypothesis(X_val)
p = predict(X_val)
a = accuracy(X_val, Y_val)
print(f'Hypothesis : {tf.squeeze(h).numpy()}')
print(f'Correct : {tf.squeeze(p).numpy()}')
print(f'Accuracy : {a*100:.6}%')
