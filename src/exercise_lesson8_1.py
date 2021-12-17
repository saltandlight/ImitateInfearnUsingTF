import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

y_data = np.array([[0],
                   [1],
                   [1], [0]], dtype=np.float32)

# X, Y는 추후 선언

W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

@tf.function
def hypo(X):
    return tf.sigmoid(tf.matmul(X, W) + b)

@tf.function
def cost(X, Y):
    hyp = hypo(X)
    return -tf.reduce_mean(Y * tf.math.log(hyp) + (1-Y) * tf.math.log(1-hyp))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 모델의 정확도 계산
@tf.function
def predict(X):
    hyp = hypo(X)
    return tf.cast(hyp > 0.5, dtype=tf.float32)

@tf.function
def accuracy(X, Y):
    pred = predict(X)
    return tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

X = tf.Variable(x_data, dtype=tf.float32)
Y = tf.Variable(y_data, dtype=tf.float32)

for step in range(10001):
    with tf.GradientTape() as tp:
        cost_val = cost(X, Y)

    train = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)
    if step % 1000 == 0:
        print(f'step = {step:06}, cost: {cost_val}, weight: {tf.squeeze(W.numpy())}')

h = hypo(X)
p = predict(X)
a = accuracy(X, Y)
print(f'Hypothesis : {tf.squeeze(h).numpy()}')
print(f'Correct : {tf.squeeze(p).numpy()}')
print(f'Accuracy: {a*100:.6}%')