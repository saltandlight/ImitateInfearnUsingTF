import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# X, Y 선언
X = tf.Variable(x_data, dtype=tf.float32)
Y = tf.Variable(y_data, dtype=tf.float32)

# W, b 선언
# W 크기 중요(x1, x2)가 in, out은 1개니까 [2, 1]
# b는 [1]
W1 = tf.Variable(tf.random.normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random.normal([10]), name='bias1')

W2 = tf.Variable(tf.random.normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random.normal([10]), name='bias2')

W3 = tf.Variable(tf.random.normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random.normal([10]), name='bias3')

W4 = tf.Variable(tf.random.normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random.normal([1]), name='bias4')

@tf.function
def layer(X):
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    return tf.sigmoid(tf.matmul(layer2, W3) + b3)

@tf.function
def hypo(X):
    layer1 = layer(X)
    return tf.sigmoid(tf.matmul(layer1, W4) + b4)

@tf.function
def cost(X, Y):
    hy_val = hypo(X)
    return -tf.reduce_mean(Y * tf.math.log(hy_val) + (1-Y) * tf.math.log(1 - hy_val))

@tf.function
def predict(X):
    hy_val = hypo(X)
    return tf.cast(hy_val > 0.5, dtype=tf.float32)

@tf.function
def accuracy(predict, Y):
    return tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
trainable_vars = [W1, b1, W2, b2, W3, b3, W4, b4]

for step in range(10001):
    with tf.GradientTape() as tp:
        cost_val = cost(X, Y)

    train = optimizer.minimize(cost_val, var_list=trainable_vars, tape=tp)
    if step % 100 == 0:
        print(f'STEP = {step}: cost = {cost_val}')

h = hypo(X)
p = predict(X)
a = accuracy(p, Y)
print(f'\nh: {h}, \np: {p},\na:{a}')
